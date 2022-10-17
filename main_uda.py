import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, evaluate_result
from utils.cluster import ClusterLoss, Normalize, BCE, NCLMemory, PairEnum
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
import mmcv
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
from einops import rearrange
import glob
import torch.nn.functional as F

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    # model parameters
    parser.add_argument('--threshold', default=0.8, type=float)
    parser.add_argument('--cluster-threshold', default=0.8, type=float)
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument("--bce-type", type=str, default='RK', help="Type of clustering techniques: cos or RK")
    parser.add_argument('--cosine-threshold', default=0.7, type=float, help='cosine similarity threshold for clustering')
    parser.add_argument('--topk', default=2, type=int, help='rank statistics threshold for clustering')
    parser.add_argument('--confidence-threshold', default=0.1, type=float, help='threshold for high-confident instance selection')

    parser.add_argument('--w-smooth', default=0.01, type=float, help='weight of smooth loss')
    parser.add_argument('--w-sparse', default=0.01, type=float, help='weight of sparse loss')
    parser.add_argument('--w-compat-u', default=1.0, type=float, help='weight of u2l loss compared to l2u')
    parser.add_argument('--w-compat', default=1.0, type=float, help='weight of compatibility loss')
    parser.add_argument('--w-cluster', default=1.0, type=float, help='weight of cluster loss')
    parser.add_argument('--w-con', default=1.0, type=float, help='weight of consistency loss')
    parser.add_argument('--w-con1', default=1.0, type=float, help='weight of clustering all consistency loss')

    parser.add_argument('--w-cls', default=1.0, type=float, help='weight of cluster anomaly score')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):

    train_data, val_data, test_data, train_loader, val_loader, test_loader, val_loader_train = build_dataloader(logger, config)
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES,
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cuda()

    criterion = ClusterLoss(config.DATA.NUM_CLASSES, args.bce_type,
        args.cosine_threshold, args.topk
    )
    # import pdb;pdb.set_trace()
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    # import pdb;pdb.set_trace()
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)

    start_epoch, best_epoch, max_auc = 0, 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    text_labels = generate_text(train_data)
    
    if config.TEST.ONLY_TEST:
        if not os.path.isdir(config.MODEL.PRETRAINED):
            # calculate statics on train set
            # out_path = config.MODEL.PRETRAINED.replace('pth','pkl').replace('ckpt_epoch','train_data')
            # epoch_data_dict = validate(val_loader_train, text_labels, model, config, out_path)
            # evaluate on val set
            out_path = config.MODEL.PRETRAINED.replace('pth','pkl')
            if os.path.exists(out_path):
                scores_dict = mmcv.load(out_path)
            else:
                scores_dict = validate(test_loader, text_labels, model, config, out_path)

            tmp_dict = {}
            for v_name in scores_dict["cls"].keys():
                # import pdb;pdb.set_trace()
                tmp = np.array(scores_dict["prd"][v_name]).copy()
                tmp_ = np.array(scores_dict["cls"][v_name]).copy()
                if tmp.shape[0] ==1:
                    tmp_dict[v_name] = [tmp[0, :] + args.w_cls * (tmp_[0,:])]
                else:
                    tmp_dict[v_name] = [tmp[:, 0] + args.w_cls * (tmp_[:, 0])] #1,32,2 np.array(scores_dict["prd"][v_name])[:,0] +
            auc_all, auc_ano = evaluate_result(tmp_dict, config.DATA.VAL_FILE)

            logger.info(f"AUC@all/ano of version {out_path.split('/')[-2]} on epoch {out_path.split('/')[-1].split('_')[-1][:-4]} : {auc_all:.4f}({auc_ano:.4f})")
            return

    if 0:
        data_dict = mmcv.load('/data4/lhui/x_clip/base/train_data_his.pkl')
        data_dict['fea'].clear()
        history_scores = []
        for key in data_dict['prd'].keys():
            history_scores.append(data_dict['prd'][key])
        his_scores = np.concatenate(history_scores,1)
        pseudo_label = np.argmax(his_scores.mean(0),-1)
        ano_idx = pseudo_label==1
        ano_scores_var = his_scores[:,ano_idx,1].var(0)
        # import pdb;pdb.set_trace()
        pseudo_threshold = np.mean(ano_scores_var,0)
        # var_sort_idx = np.argsort(ano_scores_var)
        # pseudo_threshold = ano_scores_var[var_sort_idx[int(ano_scores_var.shape[0]*args.confidence_threshold)]]
        data_dict['mask'] = {}
        data_dict['label'] = {}
        for key in data_dict['prd'].keys():
            score_var = data_dict['prd'][key][:,:,1].var(0)
            score_msk = score_var<pseudo_threshold
            data_dict['mask'][key] = score_msk
            data_dict['label'][key] = np.argmax(data_dict['prd'][key].mean(0),-1)
            # import pdb;pdb.set_trace()
    else:
        data_dict = {}
        data_dict['mask'] = {}
        data_dict['label'] = {}
        data_dict['prd'] = {}
        data_dict['cls'] = {}
        data_dict['length'] = 0

    anno_file = config.DATA.TRAIN_FILE
    vid_list = []
    with open(anno_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split()
            filename = line_split[0].split('/')[-1]
            vid_list.append(filename)

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, data_dict, vid_list)

        # calculate training statics
        out_path = os.path.join(config.OUTPUT, 'train_data_' + str(epoch) + '.pkl')
        epoch_data_dict = validate(val_loader_train, text_labels, model, config, out_path)
        # epoch_data_dict = mmcv.load(out_path)

        for key in epoch_data_dict['prd'].keys():
            if data_dict['length'] == 0:
                data_dict['prd'][key] = np.stack(epoch_data_dict['prd'][key], 1)
                data_dict['cls'][key] = np.stack(epoch_data_dict['cls'][key], 1)
                # data_dict['fea'][key] = np.stack(epoch_data_dict['fea'][key], 1)
            else:
                data_dict['prd'][key] = np.concatenate([data_dict['prd'][key], np.stack(epoch_data_dict['prd'][key], 1)], 0)
                data_dict['cls'][key] = np.concatenate([data_dict['cls'][key], np.stack(epoch_data_dict['cls'][key], 1)], 0)
                # data_dict['fea'][key] = np.concatenate([data_dict['fea'][key], np.stack(epoch_data_dict['fea'][key], 1)], 0)
        data_dict['length'] = data_dict['length'] + 1
        # import pdb;pdb.set_trace()
        if data_dict['length']>1:
            history_scores = []
            for key in data_dict['prd'].keys():
                history_scores.append(data_dict['prd'][key])
            his_scores = np.concatenate(history_scores, 1)
            pseudo_label = np.argmax(his_scores.mean(0), -1)
            ano_idx = pseudo_label == 1
            ano_scores_var = his_scores[:, ano_idx, 1].var(0)
            # import pdb;pdb.set_trace()
            pseudo_threshold = np.mean(ano_scores_var, 0)
            # var_sort_idx = np.argsort(ano_scores_var)
            # pseudo_threshold = ano_scores_var[var_sort_idx[int(ano_scores_var.shape[0]*args.confidence_threshold)]]
            for key in data_dict['prd'].keys():
                score_var = data_dict['prd'][key][:, :, 1].var(0)
                score_msk = score_var < pseudo_threshold
                data_dict['mask'][key] = score_msk
                data_dict['label'][key] = np.argmax(data_dict['prd'][key].mean(0), -1)

        # val
        out_path = os.path.join(config.OUTPUT, 'ckpt_epoch_'+str(epoch)+'.pkl')
        scores_dict = validate(val_loader, text_labels, model, config, out_path)

        tmp_dict = {}
        for v_name in scores_dict["cls"].keys():
            tmp_dict[v_name] = [scores_dict["prd"][v_name][0] + args.w_cls * scores_dict["cls"][v_name][0]]
        auc_all, auc_ano = evaluate_result(tmp_dict, config.DATA.VAL_FILE)
        is_best = auc_all > max_auc
        max_auc = max(max_auc, auc_all)
        logger.info(f"Auc of the network on epoch {epoch}: {auc_all:.4f}({auc_ano:.4f})")
        logger.info(f'Max AUC@all epoch {epoch} : {max_auc:.4f}')

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_auc, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    # config.defrost()
    # config.TEST.NUM_CLIP = 4
    # config.TEST.NUM_CROP = 3
    # config.freeze()
    # train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    # auc, auc_ano = validate(val_loader, text_labels, model, config, out_path)
    # logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {max_auc:.4f}({auc_ano:.4f})")


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, data_dict, vid_list):

    model.train()
    model.module.visual.eval()
    # model.eval()
    # model.module.ln_head.train()
    # model.module.head_max.train()
    # model.module.head_u.train()
    # model.module.prompts_visual_ln.train()
    # model.module.prompts_visual_proj.train()
    # model.module.prompts_generator.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    mil_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    cst_loss_meter = AverageMeter()
    bce_loss_meter = AverageMeter()
    cmp_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    l2norm = Normalize(2)
    bce = BCE()

    texts = text_labels.cuda(non_blocking=True)
    
    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        bz = images.shape[0]
        a_aug = images.shape[1]
        n_clips = images.shape[2]
        images = rearrange(images, 'b a k c t h w -> (b a k) t c h w')# bz*num_aug*num_clips,num_frames,h,w
        # images = images.view((-1,config.DATA.NUM_FRAMES,3)+images.size()[-2:])

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        ##read histories info

        output = model(images, texts)
        # mil loss on max scores among bags, view instance of max scores as labeled data
        logits = rearrange(output['y'], '(b a k) c -> (b a) k c', b=bz, a=a_aug, )
        scores = F.softmax(logits, dim=-1)

        scores_ano = scores[:, :, 1]
        scores_nor = scores[:, :, 0]
        max_prob_ano, max_ind = torch.max(scores_ano, dim=-1)
        max_prob_nor, _ = torch.max(scores_nor, dim=-1)

        logits_video = torch.gather(logits, 1, max_ind[:, None, None].repeat((1, 1, 2))).squeeze(1)
        margin_video = scores_ano.max(-1)[0] - scores_ano.min(-1)[0]
        max_prob_video, _ = torch.max(torch.gather(scores, 1, max_ind[:, None, None].repeat((1, 1, 2))).squeeze(1),
                                      dim=-1)
        labels_binary = label_id > 0
        loss_mil = F.cross_entropy(logits_video, labels_binary.long(), reduction='none') + \
                   F.binary_cross_entropy(margin_video, labels_binary.float(), reduction='none')
        loss_mil = loss_mil * max_prob_video
        loss_mil = loss_mil.mean()

        # pseudo loss
        logits = rearrange(logits, '(b a) k c -> b a k c', b=bz, a=a_aug, )
        logits_alt = rearrange(output['y_cluster_all'], '(b a k) c -> b a k c', b=bz, a=a_aug, )

        scores = F.softmax(logits, dim=-1)
        scores_alt = F.softmax(logits_alt, dim=-1)

        if data_dict['length'] > 1:
            vids = np.array(vid_list)[batch_data["vid"]]
            pseudo_labels = []
            masks = []
            if bz==1:
                vids = [vids]
            for ind in range(bz):
                tmp_label = data_dict['label'][vids[ind]].copy()
                tmp_mask = data_dict['mask'][vids[ind]].copy()
                if "Normal" in vids[ind]:
                    tmp_mask = np.ones_like(tmp_mask)
                tmp_ind = batch_data["frame_inds"][ind]
                tmp_ind = tmp_ind*tmp_label.shape[0]//batch_data['total_frames'][ind]
                tmp_ind = tmp_ind.reshape(n_clips, -1)[:, config.DATA.NUM_FRAMES // 2]
                pseudo_labels.append(tmp_label[tmp_ind].copy())
                masks.append(tmp_mask[tmp_ind].copy())

            pseudo_labels_np = np.stack(pseudo_labels,0)
            mask_source_np = np.stack(masks,0)
            pseudo_labels.clear()
            masks.clear()
            with torch.no_grad():
                pseudo_labels_source = torch.from_numpy(pseudo_labels_np).cuda()
                mask_source = torch.from_numpy(mask_source_np).cuda()
                pseudo_labels_source = pseudo_labels_source.view(bz,1,-1).tile((1,a_aug,1))
                mask_source = mask_source.view(bz,1,-1).tile((1,a_aug,1))
                mask_target = ~mask_source
        else:
            pseudo_labels_source = torch.max(scores,-1)[0]
            mask_target = pseudo_labels_source>=0
        # import pdb;pdb.set_trace()
        if (~mask_target).sum()==0:
            loss_ce = torch.zeros_like(loss_mil)
        else:
            nor_mask = label_id==0
            pseudo_labels_source[nor_mask.reshape(bz,-1)] = 0
            loss_ce = F.cross_entropy(logits.view(-1,2), pseudo_labels_source.view(-1), reduction='none')
            loss_ce = loss_ce * torch.max(scores,-1)[0].view(-1)
            loss_ce = loss_ce[~mask_target.view(-1)]
            loss_ce = loss_ce.mean()

        # generate target pseudo-labels, [:bz]=weak aug;[bz:]=strong aug;
        max_prob, pseudo_labels = torch.max(scores[:, 0], dim=-1)
        max_prob_alt, pseudo_labels_alt = torch.max(scores_alt[:, 0], dim=-1)

        consistency_loss = (F.cross_entropy(logits[:,1].contiguous().view(-1,2), pseudo_labels.view(-1), reduction='none') \
                                             * max_prob.view(-1).ge(args.threshold).float().detach()).mean()

        # Cluster consistency loss
        consistency_loss_alt = (F.cross_entropy(logits_alt[:,1].contiguous().view(-1,2), pseudo_labels_alt.view(-1), reduction='none') \
                                                 * max_prob_alt.view(-1).ge(args.cluster_threshold).float().detach()).mean()

        bk_feat = rearrange(output['feature_v'], '(b a k) c -> b a k c', b=bz, a=a_aug, )
        cls_nograd = rearrange(output['y_cluster_all_nograd'], '(b a k) c -> b a k c', b=bz, a=a_aug, )
        inputs = {
            "x1": bk_feat[:,0].reshape(-1,bk_feat.shape[-1]),
            "x1_norm": l2norm(bk_feat[:,0].reshape(-1,bk_feat.shape[-1])),
            "preds1_u": cls_nograd[:,0].reshape(-1,2),
            "x2": bk_feat[:,1].reshape(-1,bk_feat.shape[-1]),
            "x2_norm": l2norm(bk_feat[:,1].reshape(-1,bk_feat.shape[-1])),
            "preds2_u": cls_nograd[:,1].reshape(-1,2),
            "labels": pseudo_labels_source[:,0].reshape(-1),
            "labels_": label_id,
            "mask": ~mask_target[:,0].reshape(-1),
        }

        bce_loss, sim_matrix_all = criterion.compute_losses(inputs)
        # refine unlabel similarity matrix
        # l head compat with u with all sample
        logits_nograd = rearrange(output['y_nograd'], '(b a k) c -> b a k c', b=bz, a=a_aug, )
        p_nograd = F.softmax(logits_nograd, dim=-1)
        pairs1, _ = PairEnum(p_nograd[:,0].reshape(-1,2))
        _, pairs2 = PairEnum(p_nograd[:,1].reshape(-1,2))
        lu_compatibility_loss = bce(pairs1, pairs2, sim_matrix_all)
        # u head compat with l with source sample
        logits_alt_nograd = rearrange(output['y_cluster_all_nograd'], '(b a k) c -> b a k c', b=bz, a=a_aug, )
        p_alt_nograd = F.softmax(logits_alt_nograd, dim=-1)
        if (~mask_target).sum()>0:
            labels_s_view = pseudo_labels_source[:,0][~mask_target[:,0]].contiguous().view(-1, 1)
            sim_matrix_lb = torch.eq(labels_s_view, labels_s_view.T).float().cuda()
            sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0
            pairs1, _ = PairEnum(p_alt_nograd[:,0][~mask_target[:,0]].reshape(-1,2))
            _, pairs2 = PairEnum(p_alt_nograd[:,1][~mask_target[:,0]].reshape(-1,2))
            ul_compatibility_loss = bce(pairs1, pairs2, sim_matrix_lb.flatten())
        else:
            ul_compatibility_loss = torch.zeros_like(lu_compatibility_loss)
        compatibility_loss = lu_compatibility_loss + args.w_compat_u * ul_compatibility_loss

        scores_all = scores + args.w_cls * scores_alt
        smoothed_scores = (scores_all[:,:,1:,1] - scores_all[:,:,:-1,1])
        smoothed_loss = smoothed_scores.pow(2).sum(dim=-1).mean()

        sparsity_loss = scores_all[:,:,:,1].sum(dim=-1).mean()
        # import pdb;pdb.set_trace()
        if data_dict['length'] < 2:
            w_con = args.w_con
            w_con1 = args.w_con1
            w_cluster = 0
            w_compat = 0
            w_smooth = args.w_smooth
            w_sparse = args.w_sparse
        else:
            w_con = args.w_con
            w_con1 = args.w_con1
            w_cluster = args.w_cluster
            w_compat = args.w_compat
            w_smooth = args.w_smooth
            w_sparse = args.w_sparse

        total_loss = loss_mil + loss_ce + consistency_loss * w_con + consistency_loss_alt * w_con1 + \
                     smoothed_loss * w_smooth + sparsity_loss * w_sparse + \
                     bce_loss * w_cluster + compatibility_loss * w_compat

        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        # print(idx,total_loss)

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        mil_loss_meter.update(loss_mil.item(), len(label_id))
        ce_loss_meter.update(loss_ce.item(), len(label_id))
        cst_loss_meter.update((consistency_loss * args.w_con + consistency_loss_alt * args.w_con1).item(), len(label_id))
        bce_loss_meter.update((bce_loss * args.w_cluster).item(), len(label_id))
        cmp_loss_meter.update((compatibility_loss * args.w_compat).item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                # f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                # f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mil {mil_loss_meter.val:.4f} ({mil_loss_meter.avg:.4f})\t'
                f'ce {ce_loss_meter.val:.4f} ({ce_loss_meter.avg:.4f})\t'
                f'cst {cst_loss_meter.val:.4f} ({cst_loss_meter.avg:.4f})\t'
                f'bce {bce_loss_meter.val:.4f} ({bce_loss_meter.avg:.4f})\t'
                f'cmp {cmp_loss_meter.val:.4f} ({cmp_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(data_loader, text_labels, model, config, out_path):
    model.eval()
    vid_list = []
    if 'train' in out_path:
        anno_file = config.DATA.TRAIN_FILE
    else:
        anno_file = config.DATA.VAL_FILE

    with open(anno_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split()
            filename = line_split[0].split('/')[-1]
            vid_list.append(filename)
    # import pdb;pdb.set_trace()
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        scores_dict = dict()
        scores_dict['cls'] = dict()
        scores_dict['prd'] = dict()
        scores_dict['fea'] = dict()
        for idx, batch_data in enumerate(data_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)
            b, n, c, t, h, w = _image.size()
            _image = rearrange(_image, 'b n c t h w -> (b n) t c h w')
            # _image = _image.view(b, n, t, c, h, w)
            output = model(_image, text_inputs)

            scores_prd = F.softmax(output['y'], dim=-1)
            scores_cls = F.softmax(output['y_cluster_all'], dim=-1)

            scores_prd = rearrange(scores_prd, '(b n) c -> b n c', b=b)
            scores_np_prd = scores_prd.cpu().data.numpy().copy()
            scores_cls = rearrange(scores_cls, '(b n) c -> b n c', b=b)
            scores_np_cls = scores_cls.cpu().data.numpy().copy()
            fea = rearrange(output['feature_v'], '(b n) c -> b n c', b=b)
            fea_np = fea.cpu().data.numpy().copy()

            for ind in range(scores_np_prd.shape[0]):
                v_name = vid_list[batch_data["vid"][ind]]
                if v_name not in scores_dict['prd']:
                    scores_dict['prd'][v_name] = []
                    scores_dict['cls'][v_name] = []
                    if 'train' in out_path:
                        scores_dict['fea'][v_name] = []
                # import pdb;pdb.set_trace()
                scores_dict['prd'][v_name].append(scores_np_prd[ind])
                scores_dict['cls'][v_name].append(scores_np_cls[ind])
                if 'train' in out_path:
                    scores_dict['fea'][v_name].append(fea_np[ind])
            if idx % 500 == 0 and len(data_loader) >= 500 and 'train' not in out_path:
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Vid: {v_name}\t'
                    f'prd: [{np.max(scores_dict["prd"][v_name]):.3f}/{np.mean(scores_dict["prd"][v_name][0][:,1]):.3f}]\t'
                    f'cls: [{np.max(scores_dict["cls"][v_name]):.3f}/{np.mean(scores_dict["cls"][v_name][0][:,1]):.3f}]\t'
                )
            elif idx % 500 == 0 and len(data_loader) >= 500 and 'train' in out_path:
                logger.info(
                    f'Train: [{idx}/{len(data_loader)}]\t'
                    f'Vid: {v_name}\t'
                )
    if 'train' not in out_path:
        # import pdb;pdb.set_trace()
        auc_all_p, auc_ano_p = evaluate_result(scores_dict["prd"], config.DATA.VAL_FILE)
        auc_all_c, auc_ano_c = evaluate_result(scores_dict["cls"], config.DATA.VAL_FILE)
        # import pdb;pdb.set_trace()
        logger.info(
            # f'Test: [{idx}/{len(val_loader)}]\t'
            f'AUC_prd: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\t'
            f'AUC_cls: [{auc_all_c:.3f}/{auc_ano_c:.3f}]\t'
        )
    logger.info(f'writing results to {out_path}')
    # if not os.path.isdir(config.OUTPUT):
    #     os.mkdir(config.OUTPUT)
    mmcv.dump(scores_dict, out_path)

    # logger.info(f' * AUC@all {auc_all:.3f} AUC@ano {auc_ano:.3f}')
    return scores_dict


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)