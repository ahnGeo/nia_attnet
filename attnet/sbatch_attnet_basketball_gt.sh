#!/bin/bash

#SBATCH --job-name attnet-gt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time 2-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm-%A_%a_%x.out

python -m torch.distributed.launch --nproc_per_node 4 \
    --master_port 12330 \
    scene_parse/attr_net/tools/run_train.py \
        --dataset basketball --num_iters 50000
        --run_dir /data/ahngeo11/nia/attnet/output \
        --basketball_img_dir /local_datasets/detectron2/basketball/jpg \
        --basketball_ann_path /data/ahngeo11/nia/attnet/annotations/basketball_obj.json \
        --batch_size 10 --learning_rate 0.0016 \  # per gpu batch size, use default lr 0.002 with linear scaling
        --num_workers 8