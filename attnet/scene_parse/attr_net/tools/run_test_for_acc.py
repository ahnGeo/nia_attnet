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

att_len_list = [0, 2, 4, 8, 7, 8, 2, 3, 2]
score = torch.zeros((1, len(att_len_list[1:])))
total = 0

for data, label in train_loader :
    model.set_input(data, label)
    model.forward()
    # pred = model.get_pred()    
       
    score = torch.add(score, model.check_correct(att_len_list))
    print("score : ", score)
    total += data.shape[0]
    print("total : ", total)

print("score shape", score.shape)
score = torch.flatten(score)
print("score flatten shape", score.shape)
acc = [str(float(score[i] / total))[:8] for i in range(score.shape[0])]
print('| train acc : {}'.format(", ".join(acc)))   


score = torch.zeros((1, len(att_len_list[1:])))
total = 0

for data, label in test_loader:
    model.set_input(data, label)
    model.forward()
    # pred = model.get_pred()    #* pred.shape = (B, 2), 2 = num of target att
    
    score = torch.add(score, model.check_correct(att_len_list))
    print("score : ", score)
    total += data.shape[0]
    print("total : ", total)

print("score shape", score.shape)
score = torch.flatten(score)
print("score flatten shape", score.shape)
acc = [str(float(score[i] / total))[:8] for i in range(score.shape[0])]
print('| test acc : {}'.format(", ".join(acc)))   

