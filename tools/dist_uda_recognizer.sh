#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --master_port 47759 main_uda.py -cfg configs/ucf/32_8.yaml --output /data4/lhui/x_clip/v12.2_32_8 --accumulation-steps 8 --pretrained /data4/lhui/x_clip/v12_32_8/best.pth
# CUDA_VISIBLE_DEVICES=1 python tools/train_recognizer.py config_files/ucf/r50f8s8.py --work_dir=/data4/lhui/uda/v0 --gpus=1 --resume_from=k400_32_8.pth
# CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train_recognizer.sh config_files/ucf/r50f8s8.py 2 --work_dir=/data4/lhui/uda/v1 --resume_from=/home/lvhui/GIG/results/kinetics400_tpn_r50f8s8.pth
# CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train_recognizer.sh 2
# CUDA_VISIBLE_DEVICES=1 bash tools/dist_uda_recognizer.sh 1