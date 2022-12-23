import os
import sys
import json
import argparse
import pickle
import pycocotools.mask as mask_util
import utils 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='clevr', type=str)
parser.add_argument('--proposal_path', required=True, type=str)
parser.add_argument('--gt_scene_path', default=None, type=str)
parser.add_argument('--output_path', required=True, type=str)
parser.add_argument('--align_iou_thresh', default=0.7, type=float)
parser.add_argument('--score_thresh', default=0.9, type=float)
parser.add_argument('--suppression', default=0, type=int)
parser.add_argument('--suppression_iou_thresh', default=0.5, type=float)
parser.add_argument('--suppression_iomin_thresh', default=0.5, type=float)


def main(args):
    output_dir = os.path.dirname(args.output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
        
    scenes = None
    if args.gt_scene_path is not None:
        if args.dataset == 'clevr':
            scenes = utils.load_clevr_scenes(args.gt_scene_path)
        else:
            with open(args.gt_scene_path) as f:
                scenes = json.load(f)['scenes']#@"scenes"라는 key에 모든 정보 닮겨있음.

    with open(args.proposal_path, 'rb') as f:#! mask rcnn에서 던져준. box, mask닮겨 있다.
        proposals = pickle.load(f)
    segms = proposals['all_segms']
    boxes = proposals['all_boxes']
    nimgs = len(segms[0])
    ncats = len(segms)
    img_anns = []
    #! i: image index
    #! c: class 개수
    #! j : ??
    #! m : mask
    for i in range(nimgs):
        obj_anns = []
        for c in range(1, ncats):
            for j, m in enumerate(segms[c][i]):
                if boxes[c][i][j][4] > args.score_thresh:
                    #! 우리는 ann있으니까 여긴 안들어감.
                    if scenes is None: # no ground truth alignment
                        obj_ann = {
                            'mask': m,#ssize: [320,480], counts...
                            'image_idx': i,#비어
                            'category_idx': c,#49
                            'feature_vector': None,
                        }
                        obj_anns.append(obj_ann)
                    else:
                        mask = mask_util.decode(m)#! 마스크 추출. 클래스별 4000개 다 가져옴
                        for o in scenes[i]['objects']:
                            #! 각 이미지당 존재하는 이미지 개수 ( 이건 GT 기준임. 실제 존재하는 개수만늠 iteration)
                            mask_gt = mask_util.decode(o['mask'])#! 각 오브젝트의 GT mask.
                            if utils.iou(mask, mask_gt) > args.align_iou_thresh:
                                #! mask rcnn이 던져준 mask들과, 지금 GT에서 뽑아온 것이랑 비교해서 iou th 넘을 경우.
                                if args.dataset == 'clevr':
                                    vec = utils.get_feat_vec_clevr(o)#! CLEVR에 맞는 feature 를 뽑아줌.
                                else:
                                    vec = utils.get_feat_vec_mc(o)
                                obj_ann = { 
                                    'mask': m,
                                    'image_idx': i, 
                                    'category_idx': c,
                                    'feature_vector': vec,
                                    'score': float(boxes[c][i][j][4]),#! 얘도 위에서 th거쳐서 나온놈.
                                }
                                obj_anns.append(obj_ann)#! 한 이미지 기준 각 오브젝트 마다 실행하는 것. 
                                break
        img_anns.append(obj_anns)#! 각 이미지에 존재하는 물체들과, 각 물체들의 GT 를 저장한다.
        #@ len(img_anns)=4000
        print('| processing proposals %d / %d images' % (i+1, nimgs))

    if scenes is None and args.suppression:
        # Apply suppression on test proposals
        all_objs = []
        for i, img_ann in enumerate(img_anns):
            objs_sorted = sorted(img_ann, key=lambda k: k['score'], reverse=True)
            objs_suppressed = []
            for obj_ann in objs_sorted:
                if obj_ann['score'] > args.score_thresh:
                    duplicate = False
                    for obj_exist in objs_suppressed:
                        mo = mask_util.decode(obj_ann['mask'])
                        me = mask_util.decode(obj_exist['mask'])
                        if utils.iou(mo, me) > args.suppression_iou_thresh \
                           or utils.iomin(mo, me) > args.suppression_iomin_thresh:
                            duplicate = True
                            break
                    if not duplicate:
                        objs_suppressed.append(obj_ann)
            all_objs += objs_suppressed
            print('| running suppression %d / %d images' % (i+1, nimgs))
    else:
        all_objs = [obj_ann for img_ann in img_anns for obj_ann in img_ann]

    obj_masks = [o['mask'] for o in all_objs]
    img_ids = [o['image_idx'] for o in all_objs]
    cat_ids = [o['category_idx'] for o in all_objs]
    scores = [o['score'] for o in all_objs]
    if scenes is not None:
        feat_vecs = [o['feature_vector'] for o in all_objs]
    else:
        feat_vecs = []
    output = {
        'object_masks': obj_masks,
        'image_idxs': img_ids,
        'category_idxs': cat_ids,
        'feature_vectors': feat_vecs,
        'scores': scores,
    }
    print('| saving object annotations to %s' % args.output_path)
    with open(args.output_path, 'w') as fout:
        json.dump(output, fout)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)