#!/bin/bash

#SBATCH --job-name attnet-gt
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time 2-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm-%A_%a_%x.out

# --bs = per gpu batch size, use default lr 0.002 with linear scaling
#@ --lr = 0.002 * totalbatchsize / 50 
#@ go trainer.py, and modify index list to check acc

ANN_FILE=/data/ahngeo11/nia/attnet/annotations/basketball_obj.json
OUTPUT_DIR=/data/ahngeo11/nia/attnet/output
NUM_ITERS=25000
FEATURE_VEC_LEN=36
SPLIT_ID=15315

python -m torch.distributed.launch --nproc_per_node 4 \
    --master_port 12330 \
    scene_parse/attr_net/tools/run_train.py \
        --dataset basketball --num_iters $NUM_ITERS
        --run_dir $OUTPUT_DIR \
        --basketball_img_dir /local_datasets/detectron2/basketball/annotations/images \
        --basketball_ann_path $ANN_FILE --feature_vector_len $FEATURE_VEC_LEN \
        --batch_size 16 --learning_rate 0.0016 --num_workers 8 --val_epochs 5 --split_id $SPLIT_ID