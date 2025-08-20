# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 15:26:39 2025

@author: raufun
"""
# Gaussian filter

import numpy as np
import cv2

print("hi")

img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

h, w =img.shape

kernel = np.array([
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0]
], dtype=np.float32)  # center is at (3, 3)

# kernel = kernel / np.sum(kernel)

img_bordered = cv2.copyMakeBorder(img, 1, 3, 1, 3, cv2.BORDER_CONSTANT)

img_conv = np.zeros((h, w), dtype=np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        sum=0
        for m in range(-3, 2):
            for n in range(-3, 2):
                sum=sum+(img_bordered[i-m][j-n] * kernel[m+3][n+3])
        img_conv[i][j]=sum

# img_conv=cv2.filter2D(img_bordered, ddepth=cv2.CV_32F, kernel=kernel)

norm = np.round(cv2.normalize(img_conv, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

norm_cropped=norm[1:h+1, 1:w+1]

cv2.imshow('Original Grayscale Image', img)
cv2.imshow('Bordered Image', img_bordered)
cv2.imshow('Convolution Image', img_conv)
cv2.imshow('Normalized Image', norm)
cv2.imshow('Normalized Cropped Image', norm_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()