

import numpy as np
import cv2
import random
import math
from scipy.interpolate import UnivariateSpline

# gaussian function returns z value of x


def g(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# applies gaussian blur on image


def gaussian_blur(img, x_size, y_size, sig):

    # determines centre of the kernel
    x_centre = int((x_size - 1) / 2)
    y_centre = int((y_size - 1) / 2)

    kernel = np.zeros((y_size, x_size))

    # generates kernel
    dist_sum = 0
    for y in range(y_size):
        for x in range(x_size):

            dist = math.sqrt(((x - x_centre) ** 2) + ((y - y_centre) ** 2))

            kernel[y, x] = g(dist, 0, sig)
            dist_sum += kernel[y, x]

    kernel /= dist_sum

    # applies kernel to image amd returns blurred image
    return cv2.filter2D(img, -1, kernel)


def problem1(img, darkening_co, blending_co, mask_type):

    # darkens image by the darkening coefficient given
    def darken_image(img, darkening_co):

        dark = img * darkening_co

        return dark

    # generates rainbow mask
    def create_colour_light_mask(img):

        # creates blank  mask with same size as image
        light_mask = np.zeros((len(img[0]), len(img)), np.uint8)

        # creates points to draw shape of light leak
        a = int((211/400) * len(img))
        b = int((114/400) * len(img))
        c = int((275/400) * len(img))
        d = int((363/400) * len(img))
        e = int((215/400) * len(img))
        f = int((303/400) * len(img))
        g = int((188/400) * len(img))
        h = int((75/400) * len(img))

        # connects points and fills them
        pts = np.array([[a, b], [c, d], [e, f], [g, h]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(light_mask, [pts], 255)

        # converts from grey scale to bgr
        light_mask = cv2.cvtColor(light_mask, cv2.COLOR_GRAY2BGR)

        # creates second mask the same size as the image
        color_mask = np.zeros(img.shape, np.uint8)
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2HSV)

        # creates varying brightness pattern to apply colour
        # map onto so to create rainbow effect
        j = 0

        for i in range(len(color_mask)):

            if j < 255:
                color_mask[:, i, 2] = j
            else:
                j = 0

            j += 4

        # applies rainbow colour map to itself
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_HSV2BGR)
        color_mask = cv2.applyColorMap(color_mask, cv2.COLORMAP_RAINBOW)

        # blurs mask to give smooth transition of colours
        color_mask = gaussian_blur(color_mask, 21, 21, 20)

        # applies rainbpw colour to the shape of the light leak
        for x in range(len(light_mask)):
            for y in range(len(light_mask[0])):

                if all(light_mask[x, y]) != 0:

                    light_mask[x, y] = color_mask[x, y]

        # blurs mask to give effect of light dissapating from source
        light_mask = gaussian_blur(light_mask, 27, 27, 20)

        return light_mask

    # generates white light mask
    def create_white_light_mask(img):

        # creates blank  mask with same size as image
        mask = np.zeros((len(img), len(img[0])), np.uint8)

        # creates points to draw shape of light leak
        a = int((211/400) * len(img))
        b = int((114/400) * len(img))
        c = int((275/400) * len(img))
        d = int((363/400) * len(img))
        e = int((215/400) * len(img))
        f = int((303/400) * len(img))
        g = int((188/400) * len(img))
        h = int((75/400) * len(img))

        # connects points and fills them
        # blurs mask to give effect of light dissapating from source
        pts = np.array([[a, b], [c, d], [e, f], [g, h]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        mask = gaussian_blur(mask, 21, 21, 20)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = gaussian_blur(mask, 21, 21, 20)

        return mask

    #  alpha blending function to blend mask and darkened image
    def apply_mask(dark, mask, blending_co):

        filtered_image = ((blending_co * mask) +
                          ((1-blending_co) * dark)).astype(np.uint8)

        return filtered_image

    # CHECK ALL PARAMETERS ARE PRESENT
    if img.shape[0] <= 0 or img.shape[1] <= 0:

        raise Exception("error: invalid image")

    elif not darkening_co or (darkening_co < 0) and (darkening_co > 255):

        raise Exception("error: invalid value for darkening_co")

    elif not blending_co or (blending_co < 0) and (blending_co > 1):

        raise Exception("error: invalid value for blending_co")

    elif mask_type == 'white':

        filtered_image = apply_mask(darken_image(
            img, darkening_co), create_white_light_mask(img), blending_co)

    elif mask_type == 'rainbow':

        filtered_image = apply_mask(darken_image(
            img, darkening_co), create_colour_light_mask(img), blending_co)

    else:

        raise Exception('error: invalid parameters')

    cv2.imshow('Original Image', img)
    cv2.imshow('Filtered Image', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def problem2(img, blending_co, pencil_type):

    # creates pencil-like noise texture, takes grey scale image
    def create_pencil_strokes(img):
        prob = 0.5

        # creates blank greyscale image same size as the functions input image
        pencil_strokes = np.zeros(img.shape, np.uint8)

        thres = 1 - prob

        # randomly assigns values of 0 or 255 to each pixel to acieve salt and pepper type noise effect
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

        size = 21

        # blurs noise using motion blure to make pencil strokes

        # generates motion blur kernel kernel
        mb_kernel = np.zeros((size, size))
        mb_kernel[int((size-1)/2), :] = np.ones(size)
        mb_kernel = mb_kernel / size

        # applying the kernel to the noise texture to generate pencil effect
        pencil_strokes = cv2.filter2D(pencil_strokes, -1, mb_kernel)

        return pencil_strokes

    # determines rough outline of image and applies it to input image tto make it look more like a sketch
    def draw_edges(sketch, img):

        # canny edge detection to get image outline
        # inverts results to get black, pencil-like lines
        edges = cv2.Canny(img, 180, 200)

        # adds all images together to create sketch appearance
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if edges[i, j] == 255:
                    if not isinstance(sketch[i, j], list):
                        sketch[i, j] = 0
                    else:
                        sketch[i, j] = (0, 0, 0)

        sketch = gaussian_blur(sketch, 3, 3, 20)

        return sketch

    def grey_pencil_sketch(img, blending_co):

        # converts input image to grey scale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # creates noise texture
        pencil_strokes = create_pencil_strokes(img)

        # blends grey scale image with noise texture
        sketched_image = ((grey * blending_co) +
                          (pencil_strokes * (1 - blending_co))).astype(np.uint8)

        # draw edges onto sketch and blurs it
        sketched_image = draw_edges(sketched_image, img)
        sketched_image = gaussian_blur(sketched_image, 3, 3, 20)

        return sketched_image

    def colour_pencil_sketch(img, blending_co):

        # creates seperate noise textures for each channel
        pencil1 = create_pencil_strokes(img)
        pencil2 = create_pencil_strokes(img)

        # converts input image to grey scale and back to bgr so that all channels hold same data
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

        # splits image into seperate bgr channels
        blue_channel, green_channel, red_channel = cv2.split(img)

        # blends blue channel with first noise texture
        blue_channel = ((blue_channel * blending_co) +
                        (pencil1 * (1 - blending_co))).astype(np.uint8)

        # blends red channel with second noise texture                
        red_channel = ((red_channel * blending_co) +
                       (pencil2 * (1 - blending_co))).astype(np.uint8)

        # merges red and blue channels back together (green left blank to get purple colour)
        sketched_image = cv2.merge((blue_channel, np.zeros(
            green_channel.shape, np.uint8), red_channel))
        
        # draws edges on image and blurs it
        sketched_image = draw_edges(sketched_image, img)
        sketched_image = gaussian_blur(sketched_image, 3, 3, 20)

        return sketched_image

    if img.shape[0] <= 0 or img.shape[1] <= 0:

        raise Exception("error: invalid image")

    elif (blending_co > 1) or (blending_co < 0):

        raise Exception('error: invalid value for blending_co')

    elif pencil_type == 'grey':

        filtered_image = grey_pencil_sketch(img, blending_co)

    elif pencil_type == 'colour':

        filtered_image = colour_pencil_sketch(img, blending_co)

    else:

        raise Exception(
            'error: invalid value for pencile_type OR invalid image given')

    cv2.imshow('Original Image', img)
    cv2.imshow('Filtered Image', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def problem3(img, blur_region):

    # applies median blur to image
    def median_blur(img, blur_region):

        # creates blank image the same size as the input image
        new_img = np.zeros(img.shape,np.uint8)

        size = blur_region
        centre = int((size - 1) / 2)

        # for each pixel
        for i in range(len(img)):
            for j in range(len(img[i])):

                # create list to hold values of each colour value in neighbourhood pixels
                b_pixel_values = []
                g_pixel_values = []
                r_pixel_values = []

                # for each pixel in neighbourhood
                for x in range(-centre, centre + 1):
                    for y in range(-centre, centre + 1):
                        
                        # try to append b g and r values to appropriate list
                        # (if out of bounds, neighbourhood pixel is ignored -> avoids blank pixels)
                        try:

                            b_pixel_values.append(img[i + x, j + y, 0])
                            g_pixel_values.append(img[i + x, j + y, 1])
                            r_pixel_values.append(img[i + x, j + y, 2])
                        except:
                            continue
                
                # determines median for each colour value
                b_median = np.median(b_pixel_values)
                g_median = np.median(g_pixel_values)
                r_median = np.median(r_pixel_values)

                # formats them as bgr pixel
                pixel_value = [b_median, g_median, r_median]

                # sets the new image pixel value to be the calculate median pixel value
                new_img[i, j] = pixel_value

        return new_img

    # uses SciPy univariate spleen to generate look up table
    def create_lookup_table(x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))

    # applies look up table to image
    def apply_LUT(channel, LUT):

        for x in range(len(channel)):
            for y in range(len(channel[0])):

                channel[x, y] = LUT[channel[x, y]]

        return channel

    if img.shape[0] <= 0 or img.shape[1] <= 0:

        raise Exception("error: invalid image")

    elif (blur_region < 0) or (blur_region > len(img)/2):

        raise Exception('error: bad value for blur_region')
    
    # median blurs image
    smooth = median_blur(img, blur_region)

    # creates eperate LUT for each channel
    l_LUT = create_lookup_table([0, 64, 128, 256], [0, 50, 120, 256])
    a_LUT = create_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
    b_LUT = create_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])

    # converts image to LAB format and then splits the image into seperate channels
    l_channel, a_channel, b_channel = cv2.split(
        cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB))

    # applies relevant LUT to each channel 
    l_channel = apply_LUT(l_channel, l_LUT)
    a_channel = apply_LUT(a_channel, a_LUT)
    b_channel = apply_LUT(b_channel, b_LUT)

    #merges modified channel
    filtered_image = cv2.merge((l_channel, a_channel, b_channel))
    
    #converts image back to bgr format
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_LAB2BGR)

    cv2.imshow('Original Image', img)

    cv2.imshow('Filtered_Image', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def problem4(img, swirl_radius, swirl_intensity):

    # Performs nearest neighbour interpolation by rounding pixel coordinates to the nearest integer
    # Not used in implementation as bilinear more desirable, code remains as evidence of experimentation
    def nn_interpolation(img, i, j):

        if i < 0 or i > len(img) - 1 or j < 0 or j > len(img) - 1:

            return 0

        else:

            return img[int(round(i)), int(round(j))]

    # Performs bilinear interpolation on pixel coordinates
    def bilinear_interpolation(img, i, j):

        # rounds coordinates down to get lower coordinates
        i1 = math.floor(i)
        j1 = math.floor(j)

        # rounds coordinates down to get upper coordinates
        i2 = math.ceil(i)
        j2 = math.ceil(j)

        # if rounded coordinates are in bounds
        if i1 < 400 and i2 < 400 and j1 < 400 and j2 < 400 and i1 != i2 and j1 != j2:

            # calculate pixel values using bilinear interpolation formula (distance-weighted average of surroundig pixels)
            f_i_j1 = (((i2 - i)/(i2-i1)) *
                      img[i1, j1]) + (((i - i1)/(i2 - i1)) * img[i2, j1])
            f_i_j2 = (((i2 - i)/(i2-i1)) *
                      img[i1, j2]) + (((i - i1)/(i2 - i1)) * img[i2, j2])

            pixel_value = (((j2 - j)/(j2-j1)) * f_i_j1) + \
                (((j - j1)/(j2 - j1)) * f_i_j2)

            # rounds pixel values to integers and returns bgr values
            return [(pixel_value[0]).astype(np.uint8), (pixel_value[1]).astype(np.uint8), (pixel_value[2]).astype(np.uint8)]

        else:
            # special case for when pixels round out of range (rare) and for pixels which have not been transformed
            # to prevent division by 0 error (common)
            return img[int(round(i)), int(round(j))]

    # performs image swirl transformation

    def swirl_image(img, swirl_radius, swirl_intensity):

        # creates blank image same shape as input
        swirled_image = np.zeros(img.shape, np.uint8)

        # for each pixel in img
        for x in range(len(img)):
            for y in range(len(img[0])):

                # normalises coordinates so that they lie between -1 and 1
                i = ((2 * x) / len(img)) - 1
                j = ((2 * y) / len(img)) - 1

                # converts to their polar equivelent
                r = math.sqrt((i**2)+(j**2))
                theta = math.atan2(j, i)

                # transforms value of theta if point lies within swirl radius
                if r > 0 and r < swirl_radius:

                    theta = theta - (((swirl_radius - r) * math.pi) *
                                     (r**(1/(((swirl_radius-r) * (10**swirl_intensity))))))

                # converts point back to cartesian
                i = 0.5 * len(img) * ((r * math.cos(theta))+1)
                j = 0.5 * len(img[0]) * ((r * math.sin(theta))+1)

                # interpolates pixel value
                pixel_value = bilinear_interpolation(img, i, j)

                # reverse maps pixel to new image if in range
                if i < 0 or i > len(img) - 1 or j < 0 or j > len(img) - 1:
                    continue
                else:
                    swirled_image[x, y] = pixel_value

        return swirled_image

    def reverse_swirl(swirled_image, swirl_radius, swirl_intensity):

        # creates blank image same shape as input
        reversed_image = np.zeros(img.shape, np.uint8)

        # for each pixel in img
        for x in range(len(swirled_image)):
            for y in range(len(swirled_image[0])):

                # normalises coordinates so that they lie between -1 and 1
                i = ((2 * x) / len(img)) - 1
                j = ((2 * y) / len(img)) - 1

                # converts to their polar equivelent
                r = math.sqrt((i**2)+(j**2))
                theta = math.atan2(j, i)

                # transforms value of theta if point lies within swirl radius
                # by equal but opposite amount of original transformation
                if r > 0 and r < swirl_radius:
                    theta = theta + (((swirl_radius - r) * math.pi) *
                                     (r**(1/(((swirl_radius-r) * (10**swirl_intensity))))))

                # converts point back to cartesian
                i = 0.5 * len(img) * ((r * math.cos(theta))+1)
                j = 0.5 * len(img[0]) * ((r * math.sin(theta))+1)

                # interpolates pixel value
                pixel_value = bilinear_interpolation(swirled_image, i, j)

                # reverse maps pixel to new image if in range
                if i < 0 or i > len(img) - 1 or j < 0 or j > len(img) - 1:
                    continue
                else:
                    reversed_image[x, y] = pixel_value

        return reversed_image

    # applies simple low pass filter to the image
    def low_pass_filter(img):

        # creates kernel of size 5x5 and sets all values to 1/25
        # apllies kernel to image
        size = 5
        lp_kernel = np.ones((size, size), np.uint8)
        lp_kernel = (1 / (size**2)) * lp_kernel
        lp_filtered = cv2.filter2D(img, -1, lp_kernel)
        lp_filtered.astype(np.uint8)

        return lp_filtered

    if img.shape[0] <= 0 or img.shape[1] <= 0:

        raise Exception("error: invalid image")

    elif swirl_radius < 0 or swirl_radius > 1:

        raise Exception("error: invalid value of swirl_radius")

    elif type(swirl_intensity) != int and type(swirl_intensity) != float:

        raise Exception("error: invalid value of swirl_intensity")

    swirled_image = swirl_image(img, swirl_radius, swirl_intensity)
    swirled_image_with_lp = swirl_image(
        low_pass_filter(img), swirl_radius, swirl_intensity)
    reversed_image = reverse_swirl(
        swirled_image, swirl_radius, swirl_intensity)
    image_change = abs(img - reversed_image)

    cv2.imshow('Original Image', img)
    cv2.imshow('Swirled Image', swirled_image)
    cv2.imshow('Unswirled Image', reversed_image)
    cv2.imshow('Low Pass Swirled Image', swirled_image_with_lp)
    cv2.imshow('Image Change', image_change)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



img = cv2.imread('input1.jpg',cv2.IMREAD_COLOR)

problem4(img, 0.75, 3)