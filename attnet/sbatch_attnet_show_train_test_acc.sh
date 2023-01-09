#!/bin/bash

#SBATCH --job-name attnet-show-acc
#SBATCH -w sw14
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time 2-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o scene_parse/attr_net/tools/results/attnet-gt-480-res152-halflr-results.log

python -m torch.distributed.launch --nproc_per_node 3 --master_port 12330 \
scene_parse/attr_net/tools/run_test_for_acc.py --num_workers 16 \
--run_dir scene_parse/attr_net/tools/results \
--feature_vector_len 43 \
--dataset basketball \
--load_checkpoint_path output/480-res152-halflr-ckpt.pt \
--basketball_ann_path annotations/basketball_obj.json \
--basketball_img_dir /local_datasets/detectron2/basketball/annotations/images \
--split_id 15315 --batch_size 8 
