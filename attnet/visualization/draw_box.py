import cv2

cv2_image = cv2.imread('./A01_AA02_T026_221002_CH05_Z01_f002333.jpg', cv2.IMREAD_COLOR)
drawing_image = cv2_image.copy()
drawing_image2 = cv2_image.copy()


coords = {"x1": int(0.19349*1920 - (0.0244792*1920) // 2),
          "x2" : int(0.19349*1920 + (0.0244792*1920) // 2),
          "y1": int(0.399537*1080 - (0.0805556*1080) // 2),
          "y2" : int(0.399537*1080 + (0.0805556*1080) // 2) }

# min
# coords = {"x1": int(0.180729*1920),
#           "x2" : int(0.180729*1920 + (0.290625*1920)),
#           "y1": int(0.513889*1080),
#           "y2" : int(0.513889*1080 + (0.385185*1080)) }



# for i in range(1, 11) :
#     xi = 'x'+str(i)
#     yi = 'y'+str(i)
#     x = coords[xi]
#     y = coords[yi]
#     cv2.line(drawing_image, (x, y), (x, y), 128, 8)
#     if i == 10 :
#         continue
#     xi1 = 'x'+str(i+1)
#     yi1 = 'y'+str(i+1)
#     x1 = coords[xi1]
#     y1 = coords[yi1]
#     cv2.line(drawing_image2, (x, y), (x1, y1), 128, 8)

# cv2.line(drawing_image2, (x, y), (coords['x1'], coords['y1']), 128, 8)

# cv2.imwrite("original_img.jpg", cv2_image)
# cv2.imwrite("dot_img.jpg", drawing_image)

x1 = coords['x1']
x2 = coords['x2']
y1 = coords['y1']
y2 = coords['y2']

p1 = (x1, y1)
p2 = (x1, y2)
p3 = (x2, y2)
p4 = (x2, y1)

cv2.line(drawing_image2, p1, p2, 128, 3)
cv2.line(drawing_image2, p2, p3, 128, 3)
cv2.line(drawing_image2, p3, p4, 128, 3)
cv2.line(drawing_image2, p4, p1, 128, 3)




cv2.imwrite("0114_new_detection_yolo_box_A01_AA02_T026_221002_CH05_Z01_f002333.jpg", drawing_image2)
