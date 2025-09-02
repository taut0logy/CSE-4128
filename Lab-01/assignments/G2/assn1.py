import numpy as np
import cv2

def gaussian(u, v, sigma = 1):
    g = np.exp(-(u**2 + v**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return g

def laplacian_of_gaussian(u=5, v=5, sigma=1):
    laplacian = np.exp(-(u**2 + v**2) / (2 * sigma**2)) * (u**2 + v**2 - 2 * sigma**2) / sigma**4 * (1 / (2 * np.pi * sigma**2))
    return laplacian

def gaussian_blurr_kernel(m,sigma):
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    kernel = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            kernel[i, j] = gaussian(i - k, j - k, sigma)
    kernel /= np.sum(kernel)
    return kernel

def log_sharpen_kernel(m, sigma):
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    kernel = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            kernel[i, j] = laplacian_of_gaussian(i - k, j - k, sigma)
    return kernel

img  = cv2.imread('../../../assets/Lena.jpg', cv2.IMREAD_COLOR)

kernel_blur = gaussian_blurr_kernel(5, 1)
kernel_sharpen = log_sharpen_kernel(7, 1)

img_b, img_g, img_r =  cv2.split(img)

img_b_blur = cv2.filter2D(img_b, cv2.CV_32F, kernel=kernel_blur)
img_g_blur = cv2.filter2D(img_g, cv2.CV_32F, kernel=kernel_blur)
img_r_blur = cv2.filter2D(img_r, cv2.CV_32F, kernel=kernel_blur)

img_blur = cv2.merge((img_b_blur, img_g_blur, img_r_blur))

img_blur_norm = np.round(cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

img_b_sharpen = cv2.filter2D(img_b, cv2.CV_32F, kernel=kernel_sharpen)
img_g_sharpen = cv2.filter2D(img_g, cv2.CV_32F, kernel=kernel_sharpen)
img_r_sharpen = cv2.filter2D(img_r, cv2.CV_32F, kernel=kernel_sharpen)

img_sharpen = cv2.merge((img_b_sharpen, img_g_sharpen, img_r_sharpen))
img_sharpen_norm = np.round(cv2.normalize(img_sharpen, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

mask_b = cv2.subtract(img_b.astype(np.float32), img_b_sharpen)
mask_g = cv2.subtract(img_g.astype(np.float32), img_g_sharpen)
mask_r = cv2.subtract(img_r.astype(np.float32), img_r_sharpen)

sharp_b = cv2.add(img_b.astype(np.float32), mask_b)
sharp_g = cv2.add(img_g.astype(np.float32), mask_g)
sharp_r = cv2.add(img_r.astype(np.float32), mask_r)

sharp = cv2.merge((sharp_b, sharp_g, sharp_r))
img_sharpen_res = np.round(cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_h, img_s, img_v = cv2.split(img_hsv)

img_h_blur = cv2.filter2D(img_h, cv2.CV_32F, kernel=kernel_blur)
img_s_blur = cv2.filter2D(img_s, cv2.CV_32F, kernel=kernel_blur)
img_v_blur = cv2.filter2D(img_v, cv2.CV_32F, kernel=kernel_blur)

img_blur_2 = cv2.merge((img_h_blur, img_s_blur, img_v_blur))
img_blur_norm_2 = np.round(cv2.normalize(img_blur_2, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

img_h_sharpen = cv2.filter2D(img_h, cv2.CV_32F, kernel=kernel_sharpen)
img_s_sharpen = cv2.filter2D(img_s, cv2.CV_32F, kernel=kernel_sharpen)
img_v_sharpen = cv2.filter2D(img_v, cv2.CV_32F, kernel=kernel_sharpen)

img_sharpen_2 = cv2.merge((img_h_sharpen, img_s_sharpen, img_v_sharpen))
img_sharpen_norm_2 = np.round(cv2.normalize(img_sharpen_2, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow('Original Image (BGR)', img)

cv2.imshow('Blue Channel', img_b)
cv2.imshow('Green Channel', img_g)
cv2.imshow('Red Channel', img_r)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Blurred Blue Channel', img_b_blur)
cv2.imshow('Blurred Green Channel', img_g_blur)
cv2.imshow('Blurred Red Channel', img_r_blur)

cv2.imshow('Blurred Image (BGR)', img_blur)
cv2.imshow('Blurred Image Normalized (BGR)', img_blur_norm)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Sharpened Blue Channel', img_b_sharpen)
cv2.imshow('Sharpened Green Channel', img_g_sharpen)
cv2.imshow('Sharpened Red Channel', img_r_sharpen)

cv2.imshow('Sharpen Filtered Image (BGR)', img_sharpen)
cv2.imshow('Sharpen Filtered Image Normalized (BGR)', img_sharpen_norm)

cv2.imshow('Sharpened Image (BGR)', img_sharpen_res)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Original Image (HSV)', img_hsv)

cv2.imshow('Hue Channel', img_h)
cv2.imshow('Saturation Channel', img_s)
cv2.imshow('Value Channel', img_v)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Blurred Hue Channel', img_h_blur)
cv2.imshow('Blurred Saturation Channel', img_s_blur)
cv2.imshow('Blurred Value Channel', img_v_blur)

cv2.imshow('Blurred Image (HSV)', img_blur_2)
cv2.imshow('Blurred Image Normalized (HSV)', img_blur_norm_2)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Sharpened Hue Channel', img_h_sharpen)
cv2.imshow('Sharpened Saturation Channel', img_s_sharpen)
cv2.imshow('Sharpened Value Channel', img_v_sharpen)

cv2.imshow('Sharpened Image (HSV)', img_sharpen_2)
cv2.imshow('Sharpened Image Normalized (HSV)', img_sharpen_norm_2)

cv2.waitKey(0)
cv2.destroyAllWindows()