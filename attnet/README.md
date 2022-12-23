# nia_attnet 

annotations/make_basketball_obj_json.py : annotation 먼저 만들어 줌  
sbatch_attnet_basketball_gt.sh : train  
scene_parse/attr_net/tools/run_test_basketball_mini_situation.py :  
'''   
python tools/run_test_basketball_mini_situation.py \
--run_dir /data/ahngeo11/nia/attnet/scene_parse/attr_net/tools/results \
--dataset basketball_mini \   
--load_checkpoint_path /data/ahngeo11/nia/attnet/output/checkpoint_best.pt \
--basketball_test_ann_path /data/ahngeo11/nia/attnet/annotations/basketball_mini_obj_situation.json \
--basketball_test_img_dir /local_datasets/detectron2/basketball/jpg \
--output_path /data/ahngeo11/nia/attnet/scene_parse/attr_net/tools/results/basketball_mini_gt_train_situation.json --split_id 960 --batch_size 6  
'''  
val set에 대해 test  
scene_parse/attr_net/tools/test_results/show_test_acc.py : test_results/에 저장되는 output json 파일에서 test acc 알려 줌, 지금은 binary cls로만 되어 있어서 수정해서 사용 필요  
show_val_loss_and_acc.py : 로그에서 loss랑 acc만 모아서 보여 줌  
''' 
import json
with open("annotations/.json", 'r') as f :
	data = json.load(f)

len(data["feature_vectors"])
data["feature_vectors"].count([1, 0])  ## binary cls
data["feature_vectors"].count([0, 1])
'''  