import numpy as np
import cv2
from numpy.lib.type_check import imag 

#creates mask
def create_mask(img):
    mask = np.zeros((len(img),len(img[0])), np.uint8)
    pts = np.array([[211,114],[275,363],[215,303],[188,75]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(mask,[pts],220)
    mask = cv2.GaussianBlur(mask, (21,21),20 )

    return mask

#darkens sample image
def darken_image(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for x in range (len(img_hsv)):
        for y in range (len(img_hsv[0])):
            if (img_hsv[x,y][2] - 120) >= 0:
                img_hsv[x,y][2] -= 120
            else:
                img_hsv[x,y][2] = 0

    #returns hsv form of the image
    return img_hsv

def combine_mask(img_hsv, mask):
#applies mask to image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)

    for x in range (0, len(img_hsv)):
        for y in range (0, len(img_hsv[0])):
            if (mask_hsv[x,y][2] != 0) and ((img_hsv[x,y][2] + (mask_hsv[x,y][2] // 2)) <= 255):
                img_hsv[x,y][2] += (mask_hsv[x,y][2] // 2)
            else: continue

    filtered_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return filtered_image

#reads in sample image
img = cv2.imread('input2.jpg', cv2.IMREAD_COLOR)

#uses custom functions to apply white light streak effect 
filtered_image = combine_mask(darken_image(img), create_mask(img))

#displays generated images
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_image)

key = cv2.waitKey(0);

if (key == ord('x')):
        cv2.destroyAllWindows()
else:
    print("No image file successfully loaded.")