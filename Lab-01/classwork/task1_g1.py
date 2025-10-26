# Sobel filters (x, y)

import cv2
import numpy as np

img_lena = cv2.imread('../../assets/Lena.jpg', cv2.IMREAD_GRAYSCALE)

# Gaussian filter 5x5
gauss = np.array([
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0]
], dtype=np.float32)

image_bordered = cv2.copyMakeBorder(img_lena, 2, 2, 2, 2, cv2.BORDER_CONSTANT)

img_gauss = np.zeros(img_lena.shape, dtype=np.float32)

for i in range(img_lena.shape[0]):
    for j in range(img_lena.shape[1]):
        sum=0.0
        for m in range(-2, 3):
            for n in range(-2, 3):
                sum=sum+(image_bordered[i-m][j-n] * gauss[m+2][n+2])
        img_gauss[i][j]=sum

norm_gauss = np.round(cv2.normalize(img_gauss, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)


cv2.imshow('Original Grayscale Image', img_lena)
cv2.imshow('Gaussian Filtered Image', img_gauss)
cv2.imshow('Normalized Gaussian Filtered Image', norm_gauss)

cv2.waitKey(0)
cv2.destroyAllWindows()

img_box=cv2.imread('../../assets/box.jpg', cv2.IMREAD_GRAYSCALE)

# Prewitt filters (x, y)
px=np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
    ], dtype=np.float32)

py=np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
    ], dtype=np.float32)

img_bordered_p=cv2.copyMakeBorder(img_box, 1, 1, 1, 1, cv2.BORDER_CONSTANT)

# img_px = cv2.filter2D(img_px_bordered, ddepth=cv2.CV_32F, kernel=px)
# img_py = cv2.filter2D(img_py_bordered, ddepth=cv2.CV_32F, kernel=py)

h, w = img_box.shape

img_px=np.zeros((h,w), dtype=np.float32)
img_py=np.zeros((h,w), dtype=np.float32)

for i in range(img_box.shape[0]):
    for j in range(img_box.shape[1]):
        sumx=0.0
        sumy=0.0
        for m in range(-1, 2):
            for n in range(-1, 2):
                sumx=sumx+(img_bordered_p[i-m][j-n] * px[m+1][n+1])
                sumy=sumy+(img_bordered_p[i-m][j-n] * py[m+1][n+1])
        img_px[i][j]=sumx
        img_py[i][j]=sumy

norm_px = np.round(cv2.normalize(img_px, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
norm_py = np.round(cv2.normalize(img_py, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

grad_mag = np.sqrt(img_px**2 + img_py**2)
grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Original Grayscale Image', img_box)

cv2.imshow('Convolution Image (Prewitt x)', img_px)
cv2.imshow('Convolution Image (Prewitt y)', img_py)

cv2.imshow('Normalized Prewitt filter (Prewitt x)', norm_px)
cv2.imshow('Normalized Prewitt filter (Prewitt y)', norm_py)

cv2.imshow("Gradient magnitude", grad_mag)
cv2.imshow('Gradient magnitude normalized', grad_mag_norm)

cv2.waitKey(0)
cv2.destroyAllWindows()

