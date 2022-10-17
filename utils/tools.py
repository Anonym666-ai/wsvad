import numpy
import torch.distributed as dist
import torch
import clip
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.signal as signal

def evaluate_result(vid2abnormality, anno_file):
    UCFdata_LABEL_PATH = anno_file
    gt = []
    ans = []
    GT = []
    ANS = []
    # data_root = '/data3/UCF-Crime/UCF_Crimes/Anomaly_train_test_imgs/test/'
    video_path_list = []
    videos = {}
    for video in open(UCFdata_LABEL_PATH):
        # data_dir = data_root + video.split(' ')[0][:-4]
        # img_list = glob.glob(os.path.join(data_dir, '*.jpg'))
        vid = video.strip().split(' ')[0].split('/')[-1]
        video_len = int(video.strip().split(' ')[1])
        start_1 = int(video.split(' ')[3])
        end_1 = int(video.split(' ')[4])
        start_2 = int(video.split(' ')[5])
        end_2 = int(video.split(' ')[6])
        sub_video_gt = np.zeros((video_len,), dtype=np.int8)
        if start_1 >= 0 and end_1 >= 0:
            sub_video_gt[start_1 - 1:end_1] = 1
        if start_2 >= 0 and end_2 >= 0:
            sub_video_gt[start_2 - 1:end_2] = 1

        videos[vid] = sub_video_gt
    # import pdb;pdb.set_trace()
    for vid in videos:
        if vid not in vid2abnormality.keys():
            print("The video %s is excluded on the result!" % vid)
            continue
        # pdb.set_trace()
        cur_ab = np.array(vid2abnormality[vid])
        if cur_ab.shape[0]==1:
            cur_ab = cur_ab[0, :, 1]
        else:
            cur_ab = cur_ab[:, 0, 1]
        cur_gt = np.array(videos[vid])
        ratio = float(len(cur_gt)) / float(len(cur_ab))
        cur_ans = np.zeros_like(cur_gt, dtype='float32')
        for i in range(len(cur_ab)):
            b = int(i * ratio + 0.5)
            e = int((i + 1) * ratio + 0.5)
            # if 'Normal' in vid:
            #     cur_ans[b: e] = 0.0
            # else:
            #     cur_ans[b: e] = 1.0
            cur_ans[b: e] = cur_ab[i]
            # import pdb;pdb.set_trace()
        # cur_ans = signal.medfilt(cur_ans, kernel_size=99)
        if 'Normal' not in vid:
            gt.extend(cur_gt.tolist())
            ans.extend(cur_ans.tolist())

        GT.extend(cur_gt.tolist())
        ANS.extend(cur_ans.tolist())
        continue
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # print vid, tf_idf
        ax1.plot(cur_gt, color='r')
        ax2.plot(cur_ans, color='g')
        plt.title(vid)
        plt.show()
        root = 'png'
        plt.savefig(root + '/' + vid + '.png')
        # print('Save: ',root +'/'+vid+'.png')
        plt.close()
        # pdb.set_trace()
    # pdb.set_trace()

    # pdb.set_trace()
    # if not os.path.isdir(fpath):
    #     os.mkdir(fpath)
    # # output_file = fpath+"/%s_rgb.npz" % vid[0]

    # output_file = fpath+"/gt-ans.npz"

    ret = roc_auc_score(gt, ans)
    Ret = roc_auc_score(GT, ANS)
    # np.savez(output_file, gt=gt, ans=ans, GT=GT, ANS=ANS)
    # pdb.set_trace()
    return Ret, ret


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    
    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])

    return classes
