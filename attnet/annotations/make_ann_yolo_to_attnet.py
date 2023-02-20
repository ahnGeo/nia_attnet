import json
import os
import argparse
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument("--iou", type=float)
# args = parser.parse_args()

#* for use all attributes
attributes_path = "/data/ahngeo11/nia/attnet/annotations/basketball_attributes.json"
with open(attributes_path, 'r') as f :
    att_bind = json.load(f)
 
#* Output Json
#* Goal : dict_keys(['object_masks', 'image_idxs', 'image_name', 'feature_vectors', 'iou_scores', 'object_types'])

def get_yolo_box_coords(yolo_x, yolo_y, yolo_w, yolo_h, img_w, img_h) : 
    yolo_x *= img_w
    yolo_y *= img_h
    yolo_w *= img_w
    yolo_h *= img_h
    
    yolo_x_max = int(yolo_x + yolo_w // 2) 
    yolo_x_min = int(yolo_x - yolo_w // 2)
    yolo_y_max = int(yolo_y + yolo_h // 2) 
    yolo_y_min = int(yolo_y - yolo_h // 2)
    
    return yolo_x_max, yolo_x_min, yolo_y_max, yolo_y_min, yolo_w, yolo_h
    
def get_gt_box_coords(gt_x, gt_y, gt_w, gt_h) :
    gt_x_max = gt_x + gt_w 
    gt_x_min = gt_x  
    gt_y_max = gt_y + gt_h 
    gt_y_min = gt_y
    
    return gt_x_max, gt_x_min, gt_y_max, gt_y_min


def get_iou(detected_bbox, gt_bbox, img_w, img_h) :
    yolo_x, yolo_y, yolo_w, yolo_h = list(map(float,detected_bbox))
    yolo_x_max, yolo_x_min, yolo_y_max, yolo_y_min, yolo_w, yolo_h = get_yolo_box_coords(yolo_x, yolo_y, yolo_w, yolo_h, img_w, img_h)
    
    gt_x, gt_y, gt_w, gt_h = gt_bbox["x"], gt_bbox["y"], gt_bbox["width"], gt_bbox["height"]
    gt_x_max, gt_x_min, gt_y_max, gt_y_min = get_gt_box_coords(gt_x, gt_y, gt_w, gt_h)
    
    #* get intersection box coords
    in_x_min = max(yolo_x_min, gt_x_min)
    in_x_max = min(yolo_x_max, gt_x_max)
    in_y_min = max(yolo_y_min, gt_y_min)
    in_y_max = min(yolo_y_max, gt_y_max)
    
    #* if intersection == 0
    if in_x_min > in_x_max or in_y_min > in_y_max :
        return 0.0
    
    intersection_area = (in_x_max - in_x_min) * (in_y_max - in_y_min)
    union_area = (int(yolo_w) * int(yolo_h)) + (gt_w * gt_h) - intersection_area
    iou = float(intersection_area) / float(union_area)
    
    return iou
    
#@ set iou threshold version
# iou_threshold = args.iou

#* Load train/val set list
with open("/data/ahngeo11/nia/attnet/annotations/train_ls.txt", 'r') as f :   #@ if use mini dataset, path is "train_mini_ls.txt"
    train_list = f.readlines()
    for i in range(len(train_list)) :
        train_list[i] = train_list[i].split(".")[0]
        
with open("/data/ahngeo11/nia/attnet/annotations/val_ls.txt", 'r') as f :
    val_list = f.readlines()

json_list = train_list + val_list
root_dir = "/local_datasets/detectron2/basketball/annotations"
yolo_dir = "/local_datasets/detectron2/basketball/yolo_results"

object_masks_list = []
image_idxs_list = []
image_name_list = []
feature_vectors_list = []
iou_scores_list = []
object_types_list=[]

except_list = []


for img_id, line in enumerate(json_list) :
    
    # if img_id < 960 :   #@ basketball_mini train set num = 960
    if img_id < 15315 :   #@ basketball train set num - 15315
        split = "train"
    else :
        split = "val"
    
    line = line.strip('\n')
    
    with open(root_dir + "/{}_json/".format(split) + line + ".json", 'r') as f :
        ann_data = json.load(f)
    
    if os.path.exists(yolo_dir + "/{}/labels/".format(split) + line + ".txt") :  
        with open(yolo_dir + "/{}/labels/".format(split) + line + ".txt", 'r') as f :
            detection_data = f.readlines()
    else :
        except_list.append(line)
        continue
    
    img_obj_dict = {"player" : {"dicts" : [], "scores" : []}, "ball" : {"dicts" : [], "scores" : []}}
    
    for detected_obj in detection_data :
        if detected_obj.split()[0] == "0" :   #* player
            obj = ann_data["labelinginfo_scene representation"]["집단행동참여자"][0]
            
            if len(list(obj.values())) != 13 :    ### there are jsons with insufficient player anns
                except_list.append(line)
                continue
            
            obj_dict = {}
            
            img_w, img_h = 1920, 1080
            
            iou = get_iou(detected_obj.split()[1:], obj["location"], img_w, img_h)
            
            # if iou < iou_threshold :
            #     continue

            obj_dict["image_idxs"] = img_id
            obj_dict["image_name"] = ann_data["metaData"]["농구메타데이터"]
            obj_dict["iou_scores"] = iou
            obj_type = "player"
            obj_dict["object_types"] = "player"
            
            obj_mask = dict()
            obj_mask["size"] = [img_w, img_h]
            
            yolo_x, yolo_y, yolo_w, yolo_h = detected_obj.split()[1:]
            yolo_x, yolo_y, yolo_w, yolo_h = float(yolo_x), float(yolo_y), float(yolo_w), float(yolo_h)
            x_max, x_min, y_max, y_min, _, _ = get_yolo_box_coords(yolo_x, yolo_y, yolo_w, yolo_h, img_w, img_h) 
            
            obj_mask["counts"] = [x_max, y_max, x_min, y_min]
            obj_dict["object_masks"] = obj_mask
            
            feature_vector = [0 for j in range(43)]   #@ len of target attribute, total is 43
            
            obj_values = list(obj.values())   ### dict_keys type is not iterable

            for idx in [3, 6, 7, 8, 9, 10, 11, 12] :   #@ use total attributes about player
                if obj_values[idx] == "기타" :   
                    if idx == 7 :
                        obj_values[idx] = "선수자세기타"
                    if idx == 8 :
                        obj_values[idx] = "선수동작기타"
                feature_vector[att_bind[obj_values[idx]]] = 1    

            obj_dict["feature_vectors"] = feature_vector
            
        elif detected_obj.split()[0] == "1" :   #* ball
            obj = ann_data["labelinginfo_scene representation"]["경기도구"]
            
            if len(list(obj.values())) != 4 :  
                except_list.append(line)
                continue
            
            obj_dict = {}
            
            img_w, img_h = 1920, 1080
            
            iou = get_iou(detected_obj.split()[1:], obj["location"], img_w, img_h)

            # if iou < iou_threshold :
            #     continue
            
            obj_dict["image_idxs"] = img_id
            obj_dict["image_name"] = ann_data["metaData"]["농구메타데이터"]
            obj_dict["iou_scores"] = iou
            obj_type = "ball"
            obj_dict["object_types"] = obj_type
            
            obj_mask = dict()
            obj_mask["size"] = [1920, 1080]
            
            yolo_x, yolo_y, yolo_w, yolo_h = detected_obj.split()[1:]
            yolo_x, yolo_y, yolo_w, yolo_h = float(yolo_x), float(yolo_y), float(yolo_w), float(yolo_h)
            x_max, x_min, y_max, y_min, _, _ = get_yolo_box_coords(yolo_x, yolo_y, yolo_w, yolo_h, img_w, img_h) 

            obj_mask["counts"] = [x_max, y_max, x_min, y_min]
            obj_dict["object_masks"] = obj_mask
            
            feature_vector = [0 for j in range(43)]   #@ len of target attribute, total is 43
            
            obj_values = list(obj.values())   ### dict_keys type is not iterable

            feature_vector[att_bind[obj_values[3]]] = 1   
            obj_dict["feature_vectors"] = feature_vector
            
        else :
            continue
        
        img_obj_dict[obj_type]["dicts"].append(obj_dict)
        img_obj_dict[obj_type]["scores"].append(obj_dict["iou_scores"])
    
    for key in img_obj_dict.keys() :
        iou_list = np.array(img_obj_dict[key]["scores"])
        if len(iou_list) == 0 :
            continue
        iou_max_idx = np.argmax(iou_list)
        obj_dict = img_obj_dict[key]["dicts"][iou_max_idx]
        for each_key, each_list in zip(["object_masks", "image_idxs", "image_name", "feature_vectors", "iou_scores", "object_types"], [object_masks_list, image_idxs_list, image_name_list, feature_vectors_list, iou_scores_list, object_types_list]) :
            each_list.append(obj_dict[each_key])

                
basketball_obj_json = dict()
basketball_obj_json["object_masks"] = object_masks_list
basketball_obj_json["image_idxs"] = image_idxs_list
basketball_obj_json["image_name"] = image_name_list
basketball_obj_json["feature_vectors"] = feature_vectors_list
basketball_obj_json["iou_scores"] = iou_scores_list
basketball_obj_json["object_types"] = object_types_list

# print("IoU : ", iou_threshold)

print("number of objects : ", len(object_types_list))
print("number of ball : ", object_types_list.count("ball"))
print("number of player : ", object_types_list.count("player"))

# with open("./yolo_anns/basketball_obj_yolo_iou_{}.json".format(str(int(iou_threshold*100))), 'w') as f :   #@ output annotation file path
with open("./yolo_anns/basketball_obj_yolo_iou_{}.json".format("max"), 'w') as f :   
    json.dump(basketball_obj_json, f)
                    
# with open("/data/ahngeo11/nia/attnet/annotations/basketball_yolo_iou_50_except_ls.txt", 'w') as f :
    # f.write('\n'.join(except_list))
    

def get_iou_thresholding_info(iou) :
    with open("/data/ahngeo11/nia/attnet/annotations/train_ls.txt", 'r') as f :
        train_list = f.readlines()
        for i in range(len(train_list)) :
            train_list[i] = train_list[i].split(".")[0]
    with open("/data/ahngeo11/nia/attnet/annotations/val_ls.txt", 'r') as f :
        val_list = f.readlines()
    json_list = train_list+val_list
    for i in range(len(json_list)) :
        json_list[i] = json_list[i].strip('\n')
    
    with open("/data/ahngeo11/nia/attnet/annotations/yolo_anns/basketball_obj_yolo_iou_{}.json".format(str(iou)), 'r') as f :
            data = json.load(f)
    li_name, li_type = data["image_name"], data["object_types"]
    di = dict()
    for name in json_list :
            di[name] = []
    for name, type in zip(li_name, li_type) :
            di[name].append(type)
        
    len_li = []
    for dii in di.values() :
        len_li.append(len(dii))
    print(" 0, 1, 2, 3, 4 : \n", len_li.count(0), len_li.count(1), len_li.count(2), len_li.count(3), len_li.count(4))
        
    with open("/data/ahngeo11/nia/attnet/annotations/yolo_anns/iou_{}_info.json".format(str(iou)), 'w') as f :
            json.dump(di, f)
            
get_iou_thresholding_info("max")