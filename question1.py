import numpy as np
import cv2

#reads in sample image
img = cv2.imread('peppers.png', cv2.IMREAD_COLOR)

#darkens sample image and saves as new
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for x in range (0, len(hsv)):
    for y in range (0, len(hsv[0])):
        if (hsv[x,y][2] - 120) >= 0:
            hsv[x,y][2] -= 120
        else:
            hsv[x,y][2] = 0

dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


#displays generated images
cv2.imshow('Original Image', img)
cv2.imshow('Darkened Image', dark)


key = cv2.waitKey(0);

if (key == ord('x')):
        cv2.destroyAllWindows()
else:
    print("No image file successfully loaded.")