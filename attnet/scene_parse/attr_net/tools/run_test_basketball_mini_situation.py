import os
import json

from options import get_options
from datasets import get_dataloader
from model import get_model
import utils 


opt = get_options('test')
test_loader = get_dataloader(opt, 'test')
model = get_model(opt)

if opt.dataset == 'clevr':
    scenes = [{
        'image_index': i,
        'image_filename': 'CLEVR_val_%06d.png' % i,
        'objects': []
    } for i in range(15000)]
if opt.dataset == 'basketball_mini':
    scenes = [{
            'image_index': i,
            'image_filename': 'basketball_mini_val_%06d.jpg' % i,
            'objects': []
    } for i in range(1060)]  
if opt.dataset == "basketball" :
    scenes = [{
            'image_index': i,
            'image_filename': 'basketball_val_%06d.jpg' % i,
            'objects': []
    } for i in range(15650)]


count = 0
for data, _, idxs, cat_idxs in test_loader:
    model.set_input(data)
    model.forward()
    pred = model.get_pred()    #* pred.shape = (B, 2), 2 = num of target att
    
    for i in range(pred.shape[0]):
        img_id = idxs[i]
        obj = utils.get_attrs_basketball_mini(pred[i], 2)
        
        cid = cat_idxs[i]
        obj['gt_situation'] = "attack" if cid == 1 else "defense"
        scenes[img_id]['objects'].append(obj)
    count += idxs.size(0)
    print('%d / %d objects processed' % (count, len(test_loader.dataset)))

output = {
    'info': '%s derendered scene' % opt.dataset,
    'scenes': scenes,
}
print('| saving annotation file to %s' % opt.output_path)
utils.mkdirs(os.path.dirname(opt.output_path))
with open(opt.output_path, 'w', encoding='utf-8') as fout:
    json.dump(output, fout)