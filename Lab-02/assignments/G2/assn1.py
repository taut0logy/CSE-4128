import numpy as np
import cv2

def LoG(x, y, sigma):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)) * ((x**2 + y**2) / (2 * sigma**2) - 1) / (np.pi * sigma**4)

def LoG_kernel(m, sigma):
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    kernel = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            kernel[i, j] = LoG(i - k, j - k, sigma)
    kernel -= np.mean(kernel)
    return kernel

img = cv2.imread('../../../assets/Lena.jpg', cv2.IMREAD_GRAYSCALE)

kernel = LoG_kernel(9, 1)
img_log = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel)

zero_crossing = np.zeros_like(img_log, dtype=np.uint8)
threshold = 20

for i in range(1, img_log.shape[0] - 1):
    for j in range(1, img_log.shape[1] - 1):
        patch = img_log[i-1:i+2, j-1:j+2]
        if np.any(patch > 0) and np.any(patch < 0):
            local_region = patch.flatten()
            mean_val = np.mean(local_region)
            variance = np.mean((local_region - mean_val)**2)
            
            if variance > threshold:
                zero_crossing[i][j] = 255

cv2.imshow('Original Image', img)
cv2.imshow('LoG Filtered Image', img_log)
cv2.imshow('Robust LoG-based Edge Detection', zero_crossing)
cv2.waitKey(0)
cv2.destroyAllWindows()
