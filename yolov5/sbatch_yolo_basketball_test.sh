#!/bin/bash

#SBATCH --job-name yolo-all-test
#SBATCH -w aurora-g2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G
#SBATCH --time 1-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm/logs/%x/%A.out
#SBATCH -e slurm/logs/%x/%A.err


python detect.py \
    --weights runs/train-all/exp/weights/best.pt --source /local_datasets/nia/jpg/train --data /data/ahngeo11/nia/yolov5/data/nia_all.yaml --img 1920 --save-txt