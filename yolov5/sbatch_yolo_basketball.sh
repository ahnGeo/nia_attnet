#!/bin/bash

#SBATCH --job-name yolo-all-train
#SBATCH -w aurora-g2
#SBATCH --gres=gpu:
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=10G
#SBATCH --time 1-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm/logs/%x/%A.out
#SBATCH -e slurm/logs/%x/%A.err


python -m torch.distributed.run --nproc_per_node 2 \
    train.py --img 480 --batch 64 --epochs 300 --data nia_all.yaml --weights yolov5s.pt --device 0,1 --project runs/train-all --workers 16