# Sobel filters (x, y)

import cv2
import numpy as np

img=cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

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

img_bordered=cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT)

# img_px = cv2.filter2D(img_px_bordered, ddepth=cv2.CV_32F, kernel=px)
# img_py = cv2.filter2D(img_py_bordered, ddepth=cv2.CV_32F, kernel=py)

h, w = img.shape

img_px=np.zeros((h,w), dtype=np.float32)
img_py=np.zeros((h,w), dtype=np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        sumx=0.0
        sumy=0.0
        for m in range(-1, 2):
            for n in range(-1, 2):
                sumx=sumx+(img_bordered[i-m][j-n] * px[m+1][n+1])
                sumy=sumy+(img_bordered[i-m][j-n] * py[m+1][n+1])
        img_px[i][j]=abs(sumx)
        img_py[i][j]=abs(sumy)

norm_px = np.round(cv2.normalize(img_px, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
norm_py = np.round(cv2.normalize(img_py, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

grad_mag = np.sqrt(img_px**2 + img_py**2)
grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


cv2.imshow('Convolution Image (x)', img_px)
cv2.imshow('Convolution Image (y)', img_py)

cv2.imshow('Normalized sobel filter (x)', norm_px)
cv2.imshow('Normalized sobel filter (y)', norm_py)

cv2.imshow("Gradient magnitude", grad_mag)
cv2.imshow('Gradient magnitude normalized', grad_mag_norm)

cv2.waitKey(0)
cv2.destroyAllWindows()

