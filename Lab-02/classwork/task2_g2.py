import numpy as np
import cv2

def LoG(x, y, sigma=1):
    """Laplacian of Gaussian function"""
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)) * \
           ((x**2 + y**2) / (2 * sigma**2) - 1) / (np.pi * sigma**4)

def LoG_kernel(m, sigma):
    """Generate LoG kernel of size m x m"""
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    kernel = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            kernel[i, j] = LoG(i - k, j - k, sigma)
    kernel -= np.mean(kernel)
    return kernel

img = cv2.imread('../../assets/Lena.jpg', cv2.IMREAD_GRAYSCALE)

kernel = LoG_kernel(9, 1)
img_log = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel)

zero_crossing = np.zeros_like(img_log, dtype=np.uint8)
threshold = 20

r, c = img_log.shape
for i in range(1, r - 1):
    for j in range(1, c - 1):
        patch = img_log[i-1:i+2, j-1:j+2]

        if np.any(patch > 0) and np.any(patch < 0):
            neighbors = [img_log[i-1, j],img_log[i+1, j],img_log[i, j-1],img_log[i, j+1]]
            ZS = sum(abs(img_log[i, j] - n) for n in neighbors)

            if ZS > threshold:
                zero_crossing[i, j] = 255

cv2.imshow('Original Image', img)
cv2.imshow('LoG Filtered Image', img_log)
cv2.imshow('Zero Crossing Edge Detection', zero_crossing)
cv2.waitKey(0)
cv2.destroyAllWindows()
