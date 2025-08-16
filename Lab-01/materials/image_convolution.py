import numpy as np
import cv2

# Load grayscale image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Gaussian kernel
kernel1 = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        


'''
kernel2 = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

kernel3 = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])

'''
kernel = kernel1

h, w = img.shape

# Determine padding dynamically based on kernel size
pad_h = kernel.shape[0] // 2
pad_w = kernel.shape[1] // 2

# Add border based on kernel size
img_bordered = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

# Use float32 depth to keep values beyond 0–255
img_conv = cv2.filter2D(img_bordered, ddepth=cv2.CV_32F, kernel=kernel)

# Normalize and convert to uint8
norm = np.round(cv2.normalize(img_conv, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

# Crop the padding area
norm_cropped = norm[pad_h:h+pad_h, pad_w:w+pad_w]

# Show all images
cv2.imshow('Original Grayscale Image', img)
cv2.imshow('Bordered Image', img_bordered)
cv2.imshow('Convolution Image', img_conv)
cv2.imshow('Normalized Image', norm)
cv2.imshow('Normalized Cropped Image', norm_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
