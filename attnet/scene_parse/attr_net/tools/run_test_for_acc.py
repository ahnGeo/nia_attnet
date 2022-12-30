import os
import json

from options import get_options
from datasets import get_dataloader
from model import get_model
import utils 

import torch

opt = get_options('test')
train_loader = get_dataloader(opt, 'train')
test_loader = get_dataloader(opt, 'test')
model = get_model(opt)
model.eval_mode()
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