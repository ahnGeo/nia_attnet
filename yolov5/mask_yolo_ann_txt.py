import json

class_bind = {
    "농구공" : 1,
    "경기장" : 2,
    "3점라인" : 5,
    "페인트존" : 3,
    "골대" : 4
}

# Goal : class / x_center / y_center / width / height


def register_object(img_width, img_height, cls_id, x_center, y_center, obj_width, obj_height) :
    x = x_center / img_width
    y = y_center / img_height
    w = obj_width / img_width
    h = obj_height / img_height
    line_list = [str(cls_id), str(x), str(y), str(w), str(h)]
    line = ' '.join(line_list)
    
    return line

def register_box_object(obj_ann_list, object_dict, img_w, img_h) :
    cls_id = class_bind[object_dict["label"]]

    if type(object_dict["location"]) == list :
        for obj_location in object_dict["location"] :
            x_center = float(obj_location["x"])
            y_center = float(obj_location["y"])
            width = float(obj_location["width"])
            height = float(obj_location["height"])
            obj_ann_list.append(register_object(img_w, img_h, cls_id, x_center, y_center, width, height))       
    else :
        x_center = float(object_dict["location"]["x"])
        y_center = float(object_dict["location"]["y"])
        width = float(object_dict["location"]["width"])
        height = float(object_dict["location"]["height"])
        obj_ann_list.append(register_object(img_w, img_h, cls_id, x_center, y_center, width, height))

def register_polygon_object(obj_ann_list, object_dict, img_w, img_h) :
    cls_id = class_bind[object_dict["label"]]
    
    
    for obj_location in object_dict["location"] : ### multiple polygon type objects can be included along in one list
        state = 0
        x_list = []
        y_list = []
        
        for coord in obj_location.values() :
            if state == 0 :
                x_list.append(coord)
                state += 1
            elif state == 1 :
                y_list.append(coord)
                state -= 1
        xmin, ymin = min(x_list), min(y_list)
        xmax, ymax = max(x_list), max(y_list)
        width = float(xmax - xmin)
        height = float(ymax - ymin)
        x_center = xmin + width/2
        y_center = ymin + height/2
        
        obj_ann_list.append(register_object(img_w, img_h, cls_id, x_center, y_center, width, height))


# json_list = ["A01_AA01_T002_220916_CH01_X01_f001838.json"]
with open("json_ls.txt", 'r') as f :
    json_list = f.readlines()

root_dir = "/local_datasets/detectron2/basketball"

ball_location_empty_list = []
    
for line in json_list :
    line = line[:-1]  ### remove \n
    with open(root_dir + "/json/" + line, 'r') as f :
        data = json.load(f)
        
    obj_ann_per_img = []    
    state_not_empty = True
    
    img_w = data["imageinfo"]["width"]
    img_h = data["imageinfo"]["height"]
        
    for key in data["labelinginfo_scene representation"].keys() :
        if key == "집단행동참여자" :
            for object_dict in data["labelinginfo_scene representation"][key] :  ### [{}]
                assert object_dict["type"] == "box"
                cls_id = 0   ### category == person == 0
                x_center = float(object_dict["location"]["x"])
                y_center = float(object_dict["location"]["y"])
                width = float(object_dict["location"]["width"])
                height = float(object_dict["location"]["height"])
                obj_ann_per_img.append(register_object(img_w, img_h, cls_id, x_center, y_center, width, height))
            
        elif key == "환경" : 
            object_dict = data["labelinginfo_scene representation"]["환경"]["경기장"]
            assert object_dict["type"] == "polygon"
            register_polygon_object(obj_ann_per_img, object_dict, img_w, img_h)                                     
            
            for kkey in data["labelinginfo_scene representation"]["환경"]["경기라인"] :
                object_dict = data["labelinginfo_scene representation"]["환경"]["경기라인"][kkey]
                assert object_dict["type"] == "polygon"
                register_polygon_object(obj_ann_per_img, object_dict, img_w, img_h)    
                
        else :
            object_dict = data["labelinginfo_scene representation"][key]
            if not object_dict["location"] == {} :
                if object_dict["type"] == "box" :  
                    register_box_object(obj_ann_per_img, object_dict, img_w, img_h)                                     
                elif object_dict["type"] == "polygon" :
                    register_polygon_object(obj_ann_per_img, object_dict, img_w, img_h)                                     
            else :
                state_not_empty = False
                ball_location_empty_list.append(line)
                continue
            
    if state_not_empty == True :
        with open("/data/ahngeo11/nia/yolov5/data/annotations/basketball/" + line[:-5] + ".txt", 'w') as f :
            f.writelines("\n".join(obj_ann_per_img))
            
with open("/data/ahngeo11/nia/yolov5/data/annotations/basketball/ball_location_empty_list.txt", 'w') as f :
    f.writelines("\n".join(ball_location_empty_list))

                    
