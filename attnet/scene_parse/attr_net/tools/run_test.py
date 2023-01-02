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
elif opt.dataset == 'basketball_mini':
    scenes = [{
            'image_index': i,
            'image_filename': 'basketball_mini_val_%06d.jpg' % i,
            'objects': []
    } for i in range(1060)]  
elif opt.dataset == "basketball" :
    scenes = [{
            'image_index': i,
            'image_filename': 'basketball_val_%06d.jpg' % i,
            'objects': []
    } for i in range(15650)]


def get_attrs_basketball(feat_vec, len):
    ball_location = ["선수1", "선수2", "선수3", "선수4", "선수5", "선수6", "골대"]
    situation = ['공격', '수비']
    direction = ["정면", "왼쪽", "오른쪽", "후방"]

    obj = {
        'situation': situation[np.argmax(feat_vec[0: len])]
    }
    return obj


count = 0
for data, _, idxs, cat_idxs in test_loader:
    model.set_input(data)
    model.forward()
    pred = model.get_pred()  #* pred.shape = (B, output_dim)
    
    for i in range(pred.shape[0]):

        img_id = idxs[i]
        obj = utils.get_attrs_clevr(pred[i])  #* pred[i].shape = (1, output_dim)
        if opt.use_cat_label:
            cid = cat_idxs[i] if isinstance(cat_idxs[i], int) else cat_idxs[i].item()
            obj['color'], obj['material'], obj['shape'] = cat_dict[cid].split(' ')

        scenes[img_id]['objects'].append(obj)
    count += idxs.size(0)
    print('%d / %d objects processed' % (count, len(test_loader.dataset)))

output = {
    'info': '%s derendered scene' % opt.dataset,
    'scenes': scenes,
}
print('| saving annotation file to %s' % opt.output_path)
utils.mkdirs(os.path.dirname(opt.output_path))
with open(opt.output_path, 'w') as fout:
    json.dump(output, fout)