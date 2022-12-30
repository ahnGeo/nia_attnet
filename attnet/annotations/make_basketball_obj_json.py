import json

#* for use all attributes
att_bind = {
    "선수1" : 0,
    "선수2" : 1,
    "선수3" : 2,
    "선수4" : 3,
    "선수5" : 4,
    "선수6" : 5,
    "골대" : 6,  #공위치 - 7
    "공격" : 7,
    "수비" : 8,  #선수상황 - 2
    "정면" : 9,
    "왼쪽" : 10,
    "오른쪽" : 11,
    "후방" : 12,  #선수방향 - 4
    "서있다" : 13,
    "걷는다" : 14,
    "달리다" : 15,
    "점프하다" : 16,
    "넘어지다" : 17,
    "앉아있다" : 18,
    "미끄러지다" : 19,  
    "선수자세기타" : 20,  #선수자세 - 8
    "손으로치다" : 21,
    "한손으로 던지다" : 22,
    "두손으로 던지다" : 23,
    "손을 뻗다" : 24,
    "손으로 잡다" : 25,
    "손으로 밀다" : 26,
    "선수동작기타" : 27,  #선수동작 - 7
    "슛" : 28,
    "패스" : 29,
    "스틸" : 30,
    "드리블" : 31,
    "리바운드" : 32,
    "침투" : 33,
    "견제" : 34, 
    "해당없음" : 35, #선수행동 - 8
    "A진영" : 36,
    "B진영" : 37,  #선수위치(진영) - 2
    "3점라인밖" : 38,
    "3점라인안" : 39,
    "페인트존" : 40,   #선수위치(라인) - 3
    "A팀" : 41,
    "B팀" : 42  #선수소속 - 2
    }  

#* use total attributes for player
# att_bind = {
#     "공격" : 0,
#     "수비" : 1,  #선수상황
#     "정면" : 2,
#     "왼쪽" : 3,
#     "오른쪽" : 4,
#     "후방" : 5,  #선수방향
#     "서있다" : 6,
#     "걷는다" : 7,
#     "달리다" : 8,
#     "점프하다" : 9,
#     "넘어지다" : 10,
#     "앉아있다" : 11,
#     "미끄러지다" : 12,  
#     "선수자세기타" : 13,  #선수자세
#     "손으로치다" : 14,
#     "한손으로 던지다" : 15,
#     "두손으로 던지다" : 16,
#     "손을 뻗다" : 17,
#     "손으로 잡다" : 18,
#     "손으로 밀다" : 19,
#     "선수동작기타" : 20,  #선수동작
#     "슛" : 21,
#     "패스" : 22,
#     "스틸" : 23,
#     "드리블" : 24,
#     "리바운드" : 25,
#     "침투" : 26,
#     "견제" : 27, 
#     "해당없음" : 28, #선수행동
#     "A진영" : 29,
#     "B진영" : 30,  #선수위치(진영)
#     "3점라인밖" : 31,
#     "3점라인안" : 32,
#     "페인트존" : 33,   #선수위치(라인)
#     "A팀" : 34,
#     "B팀" : 35  #선수소속
#     }  

#@ select target attributes like this
# att_bind = {
#     "A진영" : 0,
#     "B진영" : 1  #선수위치(진영)
#     }  

#* Output Json
#* Goal : dict_keys(['object_masks', 'image_idxs', 'image_name', 'feature_vectors', 'scores'])


#* Load train/val set list
with open("/data/ahngeo11/nia/attnet/annotations/train_ls.txt", 'r') as f :   #@ if use mini dataset, path is "train_mini_ls.txt"
    train_list = f.readlines()
    for i in range(len(train_list)) :
        train_list[i] = train_list[i].split(".")[0]
        
with open("/data/ahngeo11/nia/attnet/annotations/val_ls.txt", 'r') as f :
    val_list = f.readlines()

json_list = train_list + val_list
root_dir = "/local_datasets/detectron2/basketball/annotations"

object_masks_list = []
image_idxs_list = []
image_name_list = []
feature_vectors_list = []
scores_list = []

except_list = []


for img_id, line in enumerate(json_list) :
    
    # if img_id < 960 :   #@ basketball_mini train set num = 960
    if img_id < 15315 :   #@ basketball train set num - 15315
        split = "/train_json/"
    else :
        split = "/val_json/"
    
    line = line.strip('\n')
    
    with open(root_dir + split + line + ".json", 'r') as f :
        data = json.load(f)
    
    # for i, obj in data["annotations"] :  ### 10 objects in one json file, data["ann"][0] = {dict}
    ################ sorry! attnet use only player and ball imgs
    # for i, obj in enumerate([data["labelinginfo_scene representation"]["경기도구"], data["labelinginfo_scene representation"]["집단행동참여자"][0]]) :  #@## ball and player
    for i, obj in enumerate([data["labelinginfo_scene representation"]["집단행동참여자"][0]]) :  #@## only player will be inputs
    
        if len(list(obj.values())) != 13 :    ### there are jsons with insufficient player atts
            except_list.append(line)
            continue
        
        image_idxs_list.append(img_id)
        image_name_list.append(data["metaData"]["농구메타데이터"])
        scores_list.append(100.0)
        
        obj_mask = dict()
        obj_mask["size"] = [1920, 1080]
        
        # if obj.keys()[0] == "polygon" :
        #     coords = obj["polygon"]["location"][0].values()
        #     x_coords = [ coords[j] for j in range(len(coords)) if j % 2 == 0 ]
        #     y_coords = [ coords[j] for j in range(len(coords)) if j % 2 == 1 ]
        #     obj_mask["counts"] = [max(x_coords), max(y_coords), min(x_coords), min(y_coords)]
        # elif obj.keys()[0] == "box" :
        
        #* both ball and player use box type annotation
        location_info = obj["location"]
        #*## x, y in json = xmin, ymin
        x_max = location_info["x"] + location_info["width"] 
        x_min = location_info["x"]  
        y_max = location_info["y"] + location_info["height"] 
        y_min = location_info["y"] 
        obj_mask["counts"] = [x_max, y_max, x_min, y_min]
        object_masks_list.append(obj_mask)
        
        feature_vector = [0 for j in range(43)]   #@ len of target attribute, total is 43
        
        obj_values = list(obj.values())   ### dict_keys type is not iterable
                                                        #@ only for player attributes code
                                                        #@                 0         1        2        3        4        5        6       7        8        9         10              11        12
                                                        #@ obj_values = {"type","location","선수선택","선수상황","선수성별","선수연령","선수방향","선수자세","선수동작","선수행동","선수위치(진영)","선수위치(라인)","선수소속"}
        # for idx in [10, 12] :                         #@ choose indices to select as target atts
        for idx in [3, 6, 7, 8, 9, 10, 11, 12] :                          #@ use total attributes
            if obj_values[idx] == "기타" :   
                if idx == 7 :
                    obj_values[idx] = "선수자세기타"
                if idx == 8 :
                    obj_values[idx] = "선수동작기타"
            feature_vector[att_bind[obj_values[idx]]] = 1     #* (ex) feature_vector[att_bind["공격"]] = feature_vector[7] = 1

        feature_vectors_list.append(feature_vector)

basketball_obj_json = dict()
basketball_obj_json["object_masks"] = object_masks_list
basketball_obj_json["image_idxs"] = image_idxs_list
basketball_obj_json["image_name"] = image_name_list
basketball_obj_json["feature_vectors"] = feature_vectors_list
basketball_obj_json["scores"] = scores_list
            
            
with open("/data/jong980812/nia/nia_attnet/attnet/annotations/basketball_obj.json", 'w') as f :   #@ output annotation file path
    json.dump(basketball_obj_json, f)
                    
# with open("/data/ahngeo11/nia/attnet/annotations/basketball_mini_except_ls.txt", 'w') as f :
#     f.write('\n'.join(except_list))