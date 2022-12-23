#!/bin/bash

#SBATCH --job-name yolo-basketball
#SBATCH -w sw14
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=20G
#SBATCH --time 1-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm/logs/slurm-%A_%a_%x.out
#SBATCH -e slurm/logs/slurm-%A_%a_%x.err


python -m torch.distributed.run --nproc_per_node 4 \
    train.py --img 1920 --batch 128 --epochs 300 --data nia_basketball.yaml --weights yolov5s.pt --device 0,1,2,3