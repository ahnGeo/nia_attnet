#!/bin/bash

#SBATCH --job-name attnet-gt-mini-situation
#SBATCH -w sw14
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time 2-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm-%A_%x.out
#SBATCH -e slurm-%A_%x.err


# --bs = per gpu batch size, use default lr 0.002 bs 50 with linear scaling
python -m torch.distributed.launch --nproc_per_node 4 \
    --master_port 12330 \
    scene_parse/attr_net/tools/run_train.py \
        --dataset basketball_mini --num_iters 10000 \
        --run_dir /data/ahngeo11/nia/attnet/output \
        --basketball_img_dir /local_datasets/detectron2/basketball/annotations/images \
        --basketball_ann_path /data/ahngeo11/nia/attnet/annotations/basketball_mini_obj_zone.json \
        --batch_size 8 --learning_rate 0.00128 \
        --num_workers 8 --split_id 960 --val_epochs 20 
