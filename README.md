# Image Processing Coursework


This repository contains my solutions to a 2nd year coursework assignment on Image Processing at **Durham University.**

The coursework consisted of 4 problems involving the creation of image filters implemented in Python 3.7 using OpenCV. A description of each problem is given below.

## Problem 1

 - Create a filter which applies a Light Leak and Rainbow Light Leak ('flower crown') effect.
 - Should first darken the input image using a *darkening coefficient*.
 - Should then blend the input image with a custom generated light mask using a *blending coefficient*.
 - Should accept the *darkening coefficient*, *blending coefficient* and *mode* (Standard or Rainbow) as inputs.

## Problem 2

 - Create a filter which applies a Pencil and Colour Pencil effect on an image.
 - Should generate a custom pencil effect noise texture.
 - Should blend this noise texture with greyscale version of input image.
 - In case of Colour Pencil Effect should create two distinct textures and apply them to different RGB channels.
 - Should accept the *blending coefficient* and *mode* (Monochrome or Colour) as inputs.

## Problem 3

 - Create a filter which applies a Beautification effect.
 - Should first smooth the input image.
 - Should then perform colour grading on the smoothed image image.
 - Should accept parameters that allow the level of blurring to be customised.

## Problem 4

### Part A

 - Create a filter which performs a geometric swirl on the input image.
 - Demonstrate both *Nearest Neighbour* and *Bilinear Interpolation*.
 - The filter should accept *swirl strength* and *swirl radius* as inputs.

### Part B

 - Add *Low-Pass filtering* to the filter and demonstrate its effects on anti-aliasing.

### Part C

 - Implement functionality that reverses the geometric swirl.
 - Subtract reversed image from original image to visualise difference and explain results.

The filters themselves can be found in [filters.py](https://github.com/boyla950/image_processing_coursework/blob/main/filters.py) and example uses in [examples.txt](https://github.com/boyla950/image_processing_coursework/blob/main/examples.txt).

## Report
A report was also requested. The report contains descriptions of each filter, example inputs and outputs, evaluation of their running times and the discussions/demonstrations asked for in Problem 4. The report can be found in [report.pdf](https://github.com/boyla950/image_processing_coursework/blob/main/report.pdf).

## Feedback
Full feedback for the assignment can be found in [feedback.txt](https://github.com/boyla950/image_processing_coursework/blob/main/feedback.txt). The final mark received was **87%**.

> By [boyla950](https://github.com/boyla950).
