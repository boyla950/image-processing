import numpy as np
import cv2
import random

img = cv2.imread('input1.jpg', cv2.IMREAD_COLOR)

# grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def create_pencil_strokes(img):
    prob = 0.4

    pencil_strokes = np.zeros(img.shape,np.uint8)

    thres = 1 - prob

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                pencil_strokes[i][j] = 0
            elif rdn > thres:
                pencil_strokes[i][j] = 255
            else:
                pencil_strokes[i][j] = img[i][j]
    pencil_strokes = cv2.cvtColor(pencil_strokes, cv2.COLOR_BGR2GRAY)

    size = 20

    # blurs noise mask using motion blure to make pencil strokes

    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    pencil_strokes = cv2.filter2D(pencil_strokes, -1, kernel_motion_blur)

    return pencil_strokes


def draw_edges(img):
    # uses canny edge detection to get edges within image and inverts them 
    # to make them look like pencil outline

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(grey,180,200)
    inverted_edges = np.zeros(img.shape,np.uint8)
    inverted_edges = cv2.cvtColor(inverted_edges, cv2.COLOR_BGR2GRAY)
    inverted_edges = 255 - edges

    # adds all images together to create sketch appearance
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if inverted_edges[i,j] == 0:
                grey[i,j] = 0

    grey = cv2.GaussianBlur(grey, (3,3),20 )

    return grey

grey = draw_edges(img)
pencil_strokes = create_pencil_strokes(img)

alpha = 0.6
beta = 1 - alpha

sketched_image = cv2.addWeighted(grey, alpha, pencil_strokes, beta, 0)

cv2.imshow('Original Image', img)
cv2.imshow('sketch', sketched_image)

key = cv2.waitKey(0);

if (key == ord('x')):
        cv2.destroyAllWindows()
else:
    print("No image file successfully loaded.")