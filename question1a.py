import numpy as np
import cv2 

#reads in sample image
img = cv2.imread('input1.jpg', cv2.IMREAD_COLOR)

#creates mask
mask = np.zeros((400,400), np.uint8)
pts = np.array([[211,114],[275,363],[215,303],[188,75]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.fillPoly(mask,[pts],220)
mask = cv2.GaussianBlur(mask, (21,21),20 )

#darkens sample image
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for x in range (0, len(img_hsv)):
    for y in range (0, len(img_hsv[0])):
        if (img_hsv[x,y][2] - 120) >= 0:
            img_hsv[x,y][2] -= 120
        else:
            img_hsv[x,y][2] = 0

#applies mask to image
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
mask_hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)

for x in range (0, len(img_hsv)):
    for y in range (0, len(img_hsv[0])):
        if (mask_hsv[x,y][2] != 0) and ((img_hsv[x,y][2] + (mask_hsv[x,y][2] // 2)) <= 255):
            img_hsv[x,y][2] += (mask_hsv[x,y][2] // 2)
        else: continue

filtered_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

#displays generated images
cv2.imshow('Original Image', img)
cv2.imshow('Mask', mask)
cv2.imshow('Filtered Image', filtered_image)


key = cv2.waitKey(0);

if (key == ord('x')):
        cv2.destroyAllWindows()
else:
    print("No image file successfully loaded.")