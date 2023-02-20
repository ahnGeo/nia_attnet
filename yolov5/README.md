#### data/hyps/hyp.scratch-low.yaml 
hyperparameter config, batchsize에 따라서 lr 조정 

#### data/nia_basketball.yaml  
config file  
                   
#### json_ls.txt & mask_yolo_ann_txt.py  
전체 json 리스트, mask_yolo_ann_txt.py를 실행해서 no_location_list를 만들고 나머지 이미지에 대한 yolo annotation .txt를 만듦

#### mv_no_location.sh & mv_val_split.sh 
슬랙에 공유한 no_location_list.txt에 있는 jpg들을 학습에서 제외 - mv_no_location.sh로 옮김 
A01_AA01/AA02 ... 하위 폴더별로 48개씩 val set으로 만듦 - mv_val_split.sh 

#### sbatch_yolo_basketball.sh. 
(for multi-gpus)  
(yolov5-s default == 1 gpu, batch size 64, lr 0.01 -> 4 gpus * 24 bs per gpu, lr 0.015 (linear-scaling)  
python -m torch.distributed.run --nproc_per_node 4 train.py --img 1920 --batch 96 --epochs 300 --data nia_basketball.yaml --weights yolov5s.pt --device 0,1,2,3   
