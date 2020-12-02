import numpy as np
import cv2
import random

img = cv2.imread('input1.jpg', cv2.IMREAD_COLOR)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# generates noise to act as base for pencil strokes
prob = 0.4
thres = 1 - prob

red_pencil = np.zeros(img.shape,np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rdn = random.random()
        if rdn < prob:
            red_pencil[i][j] = 0
        elif rdn > thres:
            red_pencil[i][j] = 255
        else:
            red_pencil[i][j] = img[i][j]

red_pencil = cv2.cvtColor(red_pencil, cv2.COLOR_BGR2GRAY)

blue_pencil = np.zeros(img.shape,np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rdn = random.random()
        if rdn < prob:
            blue_pencil[i][j] = 0
        elif rdn > thres:
            blue_pencil[i][j] = 255
        else:
            blue_pencil[i][j] = img[i][j]
blue_pencil = cv2.cvtColor(blue_pencil, cv2.COLOR_BGR2GRAY)

size = 30

# blurs noise mask using motion blure to make pencil strokes

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# applying the kernel to the input image
red_pencil = cv2.filter2D(red_pencil, -1, kernel_motion_blur)
blue_pencil = cv2.filter2D(blue_pencil, -1, kernel_motion_blur)
# uses canny edge detection to get edges within image and inverts them 
# to make them look like pencil outline

edges = cv2.Canny(img,180,200)
inverted_edges = np.zeros(img.shape,np.uint8)
inverted_edges = cv2.cvtColor(inverted_edges, cv2.COLOR_BGR2GRAY)
inverted_edges = 255 - edges

# adds all images together to create sketch appearance
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if inverted_edges[i,j] == 0:
            grey[i,j] = 0

grey = cv2.GaussianBlur(grey, (3,3),20 )


red_pencil_mask = np.zeros(img.shape,np.uint8)
blue_pencil_mask = np.zeros(img.shape,np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        red_pencil_mask[i,j][2] = red_pencil[i,j]
        blue_pencil_mask[i,j][0] = blue_pencil[i,j]



cv2.imshow("red", red_pencil_mask)
cv2.imshow('blue', blue_pencil_mask)


alpha = 0.5
beta = 1 - alpha
pencil_mask = cv2.addWeighted(red_pencil_mask, alpha, blue_pencil_mask, beta, 0)

grey = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
alpha = 0.6
beta = 1 - alpha
sketched_image = cv2.addWeighted(grey, alpha, pencil_mask, beta, 0)

# # making sketch coloured

cv2.imshow('Original Image', img)
cv2.imshow('sketch', sketched_image)

# cv2.imshow('coloured_sketch', coloured_sketch)

key = cv2.waitKey(0);

if (key == ord('x')):
        cv2.destroyAllWindows()
else:
    print("No image file successfully loaded.")