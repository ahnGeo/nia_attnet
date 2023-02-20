import os
import json

from options import get_options
from datasets import get_dataset
from model import get_model

from ddp import init_distributed_mode
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch

opt = get_options('test')

init_distributed_mode(opt)
train_datasets = get_dataset(opt, 'train')
test_datasets = get_dataset(opt, 'test')

cudnn.benchmark = True
global_rank=dist.get_rank()
num_tasks=dist.get_world_size()
sampler_train=torch.utils.data.DistributedSampler(
train_datasets,num_replicas=num_tasks, rank=global_rank, shuffle=True
)
sampler_test = torch.utils.data.DistributedSampler(
test_datasets, num_replicas=num_tasks, rank=global_rank, shuffle=False)

train_loader = torch.utils.data.DataLoader(
train_datasets, sampler=sampler_train,
batch_size=opt.batch_size,
num_workers=opt.num_workers,
pin_memory=True,
drop_last=True,
collate_fn=None,
)
test_loader = torch.utils.data.DataLoader(
    test_datasets, sampler=sampler_test,
    batch_size=opt.batch_size,
    num_workers=opt.num_workers,
    pin_memory=True,
    drop_last=False
)

total_batch_size = opt.batch_size  * dist.get_world_size()
print(f"total_batch_size is {total_batch_size}")
print(f"Opt \n\n {opt}")

model = get_model(opt)
device = torch.device(opt.gpu)
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
model = model.module

model.eval()

att_len_list = [0, 7, 2, 4, 8, 7, 8, 2, 3, 2]
score = torch.zeros((1, len(att_len_list[1:])))
total = 0
score_total = 0

for data, label in train_loader :
    model.set_input(data, label)
    model.forward()
    # pred = model.get_pred()    
    
    for label_each, pred_each in zip(label, model.pred.cpu()) :
        score_each, start_idx, end_idx = 0, 0, 0
        for i in range(len(att_len_list[1:])) :
            start_idx += att_len_list[i]
            end_idx += att_len_list[i + 1]
            if torch.argmax(label_each[start_idx : end_idx]) == torch.argmax(pred_each[start_idx : end_idx]) :
                score_each += 1
        if score_each == len(att_len_list[1:]) :
            score_total += 1 
       
    score = torch.add(score, model.check_correct(att_len_list))
    print("score : ", score)
    total += data.shape[0]
    print("score total : ", score_total)

score = torch.flatten(score)
acc = [str(float(score[i] / total))[:8] for i in range(score.shape[0])]
print("total", total)
print('| train acc per category : {}'.format(", ".join(acc)))   
print('| train acc total : {}'.format(score_total / total)) 


score = torch.zeros((1, len(att_len_list[1:])))
total = 0
score_total = 0


for data, label in test_loader:
    model.set_input(data, label)
    model.forward()
    
    for label_each, pred_each in zip(label, model.pred.cpu()) :
        score_each, start_idx, end_idx = 0, 0, 0
        for i in range(len(att_len_list[1:])) :
            start_idx += att_len_list[i]
            end_idx += att_len_list[i + 1]
            if torch.argmax(label_each[start_idx : end_idx]) == torch.argmax(pred_each[start_idx : end_idx]) :
                score_each += 1
        if score_each == len(att_len_list[1:]) :
            score_total += 1 
    
    score = torch.add(score, model.check_correct(att_len_list))
    print("score : ", score)
    print("score total : ", score_total)
    total += data.shape[0]
    
score = torch.flatten(score)
acc = [str(float(score[i] / total))[:8] for i in range(score.shape[0])]
print("total", total)
print('| test acc per category : {}'.format(", ".join(acc))) 
print('| test acc total : {}'.format(score_total / total))   
  
 
#^ 'total' can be more than dataset's len in setting using both ball and player objects

exit(0)