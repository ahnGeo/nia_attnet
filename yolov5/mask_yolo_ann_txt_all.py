import json

class_bind = {
    "선수" : 0,
    "공" : 1,
    "경기장" : 2,
    "3점라인" : 5,
    "페인트존" : 3,
    "골대" : 4,
    "전위" : 6,
    "후위" : 7,
    "네트" : 8,
    "골에어리어" : 9,
    "1루베이스" : 10,
    "2루베이스" : 11,
    "3루베이스" : 12,
    "홈베이스" : 13,
    "마운드" : 14,
    "타석" : 15,
    "내야" : 16,
    "1루심" : 17,
    "3루심" : 18,
    "구심" : 19,
    "코너에어리어" : 20,
    "페널티에어리어" : 21,
    "페널티마크" : 22,
    "페널티아크" : 23   
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

def register_box_object(obj_ann_list, object_dict, img_w, img_h, cls_id=None) :
    if cls_id == None :
        cls_id = class_bind[object_dict["label"]]

    if type(object_dict["location"]) == list :
        for obj_location in object_dict["location"] :
            x_min = float(obj_location["x"])
            y_min = float(obj_location["y"])
            width = float(obj_location["width"])
            height = float(obj_location["height"])
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            obj_ann_list.append(register_object(img_w, img_h, cls_id, x_center, y_center, width, height))       
    else :
        obj_location = object_dict["location"]
        x_min = float(obj_location["x"])
        y_min = float(obj_location["y"])
        width = float(obj_location["width"])
        height = float(obj_location["height"])
        x_center = x_min + width / 2
        y_center = y_min + height / 2
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
with open("/data/ahngeo11/nia/yolov5/new_json_ls.txt", 'r') as f :
    json_list = f.readlines()

root_dir = "/local_datasets/nia"

ball_location_empty_list = []
    
for line in json_list :
    line = line.strip('\n')  ### remove \n
    with open(root_dir + "/json/" + line, 'r') as f :
        data = json.load(f)
        
    print(line)
        
    obj_ann_per_img = []    
    state_not_empty = True
    
    img_w = data["imageinfo"]["width"]
    img_h = data["imageinfo"]["height"]
          
    for obj in data["annotation"] :
        object_dict = obj[list(obj.keys())[0]]
        if not object_dict["location"] == [] :
            if object_dict["type"] == "box" :  
                register_box_object(obj_ann_per_img, object_dict, img_w, img_h)                                     
            elif object_dict["type"] == "polygon" :
                register_polygon_object(obj_ann_per_img, object_dict, img_w, img_h)                                     
        else :
            state_not_empty = False
            ball_location_empty_list.append(line)
            continue
        
            
    if state_not_empty == True :
        with open("/local_datasets/nia/jpg/" + line[:-5] + ".txt", 'w') as f :
            f.writelines("\n".join(obj_ann_per_img))
            
with open("/data/ahngeo11/nia/yolov5/data/annotations/all/location_empty_list.txt", 'w') as f :
    f.writelines("\n".join(ball_location_empty_list))

                    
