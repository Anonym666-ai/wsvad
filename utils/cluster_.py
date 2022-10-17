import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np


class NCLMemory(nn.Module):
    """Memory Module for NCL"""
    def __init__(self, inputSize, K=2000, T=0.05, num_class=5, knn=None, w_pos=0.2, hard_iter=5, num_hard=400):
        super(NCLMemory, self).__init__()
        self.inputSize = inputSize  # feature dim
        self.queueSize = K  # memory size
        self.T = T
        self.index = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_class = num_class
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.w_pos = w_pos
        self.hard_iter = hard_iter
        self.num_hard = num_hard
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

        self.criterion = nn.CrossEntropyLoss()
        # number of positive
        if knn == -1:
            # default set
            self.knn = int(self.queueSize / num_class / 2)
        else:
            self.knn = knn

        # label for the labeled data
        self.label = nn.Parameter(torch.zeros(self.queueSize) - 1)
        self.label.requires_grad = False

    def forward(self, q, k, labels=None, hard_negative=False, labeled=False, la_memory=None):
        batchSize = q.shape[0]
        self.k_no_detach = k
        k = k.detach()
        self.feat = q
        self.this_labels = labels
        self.k = k.detach()
        self.la_memory = la_memory

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        x = out
        x = x.squeeze()
        if labeled:
            loss = self.supervised_loss(x, self.label, labels)
        else:
            loss = self.ncl_loss(x, hard_negative)

        # update memory
        self.update_memory(batchSize, q, labels)

        return loss

    def supervised_loss(self, inputs, all_labels, la_labels):
        targets_onehot = torch.zeros(inputs.size()).to(self.device)
        for i in range(inputs.size(0)):
            this_idx = all_labels == la_labels[i].float()
            one_tensor = torch.ones(1).to(self.device)
            this_idx = torch.cat((one_tensor == 1, this_idx))
            ones_mat = torch.ones(torch.nonzero(this_idx).size(0)).to(self.device)
            weights = F.softmax(ones_mat, dim=0)
            targets_onehot[i, this_idx] = weights
        # targets_onehot[:, 0] = 0.2
        targets = targets_onehot.detach()
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def ncl_loss(self, inputs, hard_negative):
        targets = self.smooth_hot(inputs.detach().clone())
        if not hard_negative:
            outputs = F.log_softmax(inputs, dim=1)
            loss = - (targets * outputs)
            loss = loss.sum(dim=1)
            loss = loss.mean(dim=0)
            return loss
        else:
            loss = self.ncl_hng_loss(self.feat, inputs, targets, self.memory.clone())
            return loss

    def smooth_hot(self, inputs):
        # Sort
        value_sorted, index_sorted = torch.sort(inputs[:, :], dim=1, descending=True)
        ldb = self.w_pos
        ones_mat = torch.ones(inputs.size(0), self.knn).to(self.device)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)

        weights = F.softmax(ones_mat, dim=1) * (1 - ldb)
        targets_onehot.scatter_(1, index_sorted[:, 0:self.knn], weights)
        targets_onehot[:, 0] = float(ldb)

        return targets_onehot

    def ncl_hng_loss(self, feat, inputs, targets, memory):
        new_simi = []
        new_targets = []

        _, index_sorted_all = torch.sort(inputs[:, 1:], dim=1, descending=True)  # ignore first self-similarity
        _, index_sorted_all_all = torch.sort(inputs, dim=1, descending=True)  # consider all similarities

        if self.num_class == 5:
            num_neg = 50
        else:
            num_neg = 400

        for i in range(feat.size(0)):
            neg_idx = index_sorted_all[i, -num_neg:]
            la_memory = self.la_memory.detach().clone()
            neg_memory = memory[neg_idx].detach().clone()

            # randomly generate negative features
            new_neg_memory = []
            for j in range(self.hard_iter):
                rand_idx = torch.randperm(la_memory.size(0))
                this_new_neg_memory = (neg_memory * 1 + la_memory[rand_idx][:num_neg] * 2) / 3
                new_neg_memory.append(this_new_neg_memory)
                this_new_neg_memory = (neg_memory * 2 + la_memory[rand_idx][:num_neg] * 1) / 3
                new_neg_memory.append(this_new_neg_memory)
            new_neg_memory = torch.cat(new_neg_memory, dim=0)
            new_neg_memory = F.normalize(new_neg_memory)

            # select hard negative samples
            this_neg_simi = feat[i].view(1, -1).mm(new_neg_memory.t())
            value_sorted, index_sorted = torch.sort(this_neg_simi.view(-1), dim=-1, descending=True)
            this_neg_simi = this_neg_simi[0, index_sorted[:self.num_hard]]
            this_neg_simi = this_neg_simi / self.T

            targets_onehot = torch.zeros(this_neg_simi.size()).to(self.device)
            this_simi = torch.cat((inputs[i, index_sorted_all_all[i, :]].view(1, -1),
                                   this_neg_simi.view(1, -1)), dim=1)
            this_targets = torch.cat((targets[i, index_sorted_all_all[i, :]].view(1, -1),
                                      targets_onehot.view(1, -1)), dim=1)

            new_simi.append(this_simi)
            new_targets.append(this_targets)

        new_simi = torch.cat(new_simi, dim=0)
        new_targets = torch.cat(new_targets, dim=0)

        outputs = F.log_softmax(new_simi, dim=1)
        loss = - (new_targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)

        return loss

    def update_memory(self, batchSize, k, labels):
        # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)

            if labels is not None:
                self.label.index_copy_(0, out_ids, labels.float().detach().clone())

            self.index = (self.index + batchSize) % self.queueSize

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


class ClusterLoss():
    def __init__(self, num_classes, bce_type, cosine_threshold, topk):
        # super(NCLMemory, self).__init__()
        self.num_classes = num_classes
        self.bce_type = bce_type
        self.costhre = cosine_threshold
        self.topk = topk
        self.bce = BCE()

    def compute_losses(self, inputs, include_label=False, unlabel_only=False):
        assert (include_label == False) or (unlabel_only == False)
        bce_loss = 0.0
        feat, feat_q, output2 = \
            inputs["x1"], inputs["x1_norm"], inputs["preds1_u"]
        feat_bar, feat_k, output2_bar = \
            inputs["x2"], inputs["x2_norm"], inputs["preds2_u"]
        label = inputs["labels"]

        if unlabel_only:
            mask_lb = inputs["mask"]
        else:
            mask_lb = torch.zeros_like(inputs["mask"]).bool()
        
        prob2, prob2_bar = F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)

        rank_feat = (feat[~mask_lb]).detach()
        if self.bce_type == 'cos':
            # default: cosine similarity with threshold
            feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
            tmp_distance_ori = torch.bmm(
                feat_row.view(feat_row.size(0), 1, -1),
                feat_col.view(feat_row.size(0), -1, 1)
            )
            tmp_distance_ori = tmp_distance_ori.squeeze()
            target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
            target_ulb[tmp_distance_ori > self.costhre] = 1
        elif self.bce_type == 'RK':
            # top-k rank statics
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :self.topk], rank_idx2[:, :self.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().cuda()
            target_ulb[rank_diff > 0] = -1

        if include_label:
            # use source domain label for similar/dissimilar
            labels = labels_s.contiguous().view(-1, 1)
            mask_l = torch.eq(labels, labels.T).float().to(device)
            mask_l = (mask_l - 0.5) * 2.0
            target_ulb_t = target_ulb.view(feat.size(0), -1)
            target_ulb_t[:num_s, :num_s] = mask_l
            target_ulb = target_ulb_t.flatten()

        prob1_ulb, _ = PairEnum(prob2[~mask_lb])
        _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])
        # if 1 in label:
        # import pdb;pdb.set_trace()
        bce_loss = self.bce(prob1_ulb, prob2_ulb, target_ulb)
        return bce_loss, target_ulb

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out