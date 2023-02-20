#!/bin/bash

#SBATCH --job-name yolo-basketball-test
#SBATCH -w sw14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem-per-gpu=20G
#SBATCH --time 1-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm/logs/%A_%x.out
#SBATCH -e slurm/logs/%A_%x.err


python detect.py --weights /data/ahngeo11/nia/yolov5/runs/train_with_true_right_coords/exp4/weights/best.pt \
--source /local_datasets/detectron2/basketball/jpg/val/ \
--data /data/ahngeo11/nia/yolov5/data/nia_basketball.yaml \
--img 1920 \
--save-txt