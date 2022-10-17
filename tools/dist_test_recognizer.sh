#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_0.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_1.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_2.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_3.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_4.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_5.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_6.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_7.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_8.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_9.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_10.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_11pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_12.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_13.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_14.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_15.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_16.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_17.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_18.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_19.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_20.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_21.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_22.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_23.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_24.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_25.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_26.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_27.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v10 --only_test --pretrained /data4/lhui/x_clip/v10/ckpt_epoch_28.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 4777 main.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v9 --only_test --pretrained /data4/lhui/x_clip/v9/ckpt_epoch_29.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 47771 main_.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v3.2 --only_test --pretrained /data4/lhui/x_clip/v3.2/ckpt_epoch_6.pth
$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 47774 main_uda.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v11.10 --only_test --pretrained /data4/lhui/x_clip/v11.10/ckpt_epoch_9.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 47772 main_.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v4 --only_test --pretrained /data4/lhui/x_clip/v4/ckpt_epoch_4.pth
#$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 47773 main_.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v7.2 --only_test --pretrained /data4/lhui/x_clip/v7.2/ckpt_epoch_7.pth


#CUDA_VISIBLE_DEVICES=1 python tools/train_recognizer.py config_files/ucf/r50f8s8.py --work_dir=/data4/lhui/uda/v0 --gpus=1 --resume_from=/home/lvhui/GIG/results/kinetics400_tpn_r50f8s8.pth

# CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train_recognizer.sh config_files/ucf/r50f8s8.py 2 --work_dir=/data4/lhui/uda/v1 --resume_from=/home/lvhui/GIG/results/kinetics400_tpn_r50f8s8.pth

# CUDA_VISIBLE_DEVICES=1 bash tools/dist_test_recognizer.sh 1