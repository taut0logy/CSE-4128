# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:26:21 2025

@author: NLP_LAB
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../../assets/histogram.jpg', cv2.IMREAD_GRAYSCALE)

L = 256

h,w = img.shape

size = h*w

hist = cv2.calcHist([img],[0],None,[256],[0,256])

hist = cv2.transpose(hist)[0]

pdf = hist / size

cdf = np.zeros(256)

cdf[0] = pdf[0]

for i in range(1, 256):
    cdf[i] = cdf[i-1] + pdf[i]

trn = np.zeros(256, dtype = np.uint8)

for i in range(256):
    trn[i] = (cdf[i] * (L-1))

img_eq =  cv2.LUT(img, trn)

hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0,256])
hist_eq = cv2.transpose(hist_eq)[0]

pdf_eq = hist_eq / size
cdf_eq = np.zeros(256)
cdf_eq[0] = pdf_eq[0]
for i in range(1, 256):
    cdf_eq[i] = cdf_eq[i-1] + pdf_eq[i]

hist_eq_f = cv2.equalizeHist(img)

hist_eq_f_val = cv2.calcHist([hist_eq_f], [0], None, [256], [0,256])
hist_eq_f_val = cv2.transpose(hist_eq_f_val)[0]

plt.figure(figsize=(18, 24))

plt.subplot(4,2,1)
plt.imshow(hist_eq_f, cmap='gray')
plt.title('Equalized Image (built-in)')
plt.axis("off")

plt.subplot(4,2,2)
plt.title('Equalized Histogram (built-in)')
plt.grid()
plt.bar(range(256), hist_eq_f_val, color='green')

plt.subplot(4,2,3)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(4,2,4)
plt.imshow(img_eq, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(4,2,5)
plt.title('Original Histogram')
plt.grid()
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.bar(range(256), hist, color = 'blue')

plt.subplot(4,2,6)
plt.title('Equalized Histogram')
plt.grid()
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.bar(range(256), hist_eq, color = 'blue')

plt.subplot(4,2,7)
plt.title('Original CDF')
plt.grid()
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Distribution')
plt.plot(range(256), cdf, color = 'red')

plt.subplot(4,2,8)
plt.title('Equalized CDF')
plt.grid()
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Distribution')
plt.plot(range(256), cdf_eq, color = 'red')

plt.tight_layout()

plt.show()