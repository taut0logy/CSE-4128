import cv2
import numpy as np
from kernels import gaussian_blurr_kernel, laplacian_of_gaussian_kernel, gaussian_derivative_kernel_first

img_bw=cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
img_bgr=cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)

img_b, img_g, img_r = cv2.split(img_bgr)

# b=img_color.copy()
# g=img_color.copy()
# r=img_color.copy()

# b[:,:,1] = 0
# b[:,:,2] = 0

# g[:,:,0] = 0
# g[:,:,2] = 0

# r[:,:,0] = 0
# r[:,:,1] = 0


kernel = gaussian_blurr_kernel(5, 1)
kernel2 = laplacian_of_gaussian_kernel(7, 1)

kernel3 = np.array([
    [0,-1, 0],
    [-1, 4, -1],
    [0,-1, 0]
], dtype=np.float32)

kernel4 = np.array([
    [-1,-1, -1],
    [-1, 8, -1],
    [-1,-1, -1]
], dtype=np.float32)

kernel5x, kernel5y = gaussian_derivative_kernel_first(5, 1)

img_gauss_conv_bw = cv2.filter2D(img_bw, ddepth=cv2.CV_32F, kernel=kernel)
img_gauss_norm_bw = cv2.normalize(img_gauss_conv_bw, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

img_dog_x_conv_bw = cv2.filter2D(img_bw, ddepth=cv2.CV_32F, kernel=kernel5x)
img_dog_y_conv_bw = cv2.filter2D(img_bw, ddepth=cv2.CV_32F, kernel=kernel5y)
img_dog_mag_bw = np.sqrt(img_dog_x_conv_bw**2 + img_dog_y_conv_bw**2)
img_dog_norm_bw = cv2.normalize(img_dog_mag_bw, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

img_lap4_conv_bw = cv2.filter2D(img_bw, ddepth=cv2.CV_32F, kernel=kernel3)
img_lap4_norm_bw = cv2.normalize(img_lap4_conv_bw, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

img_lap8_conv_bw = cv2.filter2D(img_bw, ddepth=cv2.CV_32F, kernel=kernel4)
img_lap8_norm_bw = cv2.normalize(img_lap8_conv_bw, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

img_log_conv_bw = cv2.filter2D(img_bw, ddepth=cv2.CV_32F, kernel=kernel2)
img_log_norm_bw = cv2.normalize(img_log_conv_bw, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

b_conv_gauss = cv2.filter2D(img_b, ddepth=cv2.CV_32F, kernel=kernel)
g_conv_gauss = cv2.filter2D(img_g, ddepth=cv2.CV_32F, kernel=kernel)
r_conv_gauss = cv2.filter2D(img_r, ddepth=cv2.CV_32F, kernel=kernel)

img_gauss_conv = cv2.merge((b_conv_gauss, g_conv_gauss, r_conv_gauss))
img_gauss_norm = cv2.normalize(img_gauss_conv, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

b_conv_log = cv2.filter2D(img_b, ddepth=cv2.CV_32F, kernel=kernel2)
g_conv_log = cv2.filter2D(img_g, ddepth=cv2.CV_32F, kernel=kernel2)
r_conv_log = cv2.filter2D(img_r, ddepth=cv2.CV_32F, kernel=kernel2)

img_log_conv = cv2.merge((b_conv_log, g_conv_log, r_conv_log))
img_log_norm = cv2.normalize(img_log_conv, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# img_gauss_res_bw = cv2.GaussianBlur(img_bw, (15, 15), 3)
# img_gauss_res_norm_bw = cv2.normalize(img_gauss_res_bw, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# img_lap_res_bw = cv2.Laplacian(img_bw, cv2.CV_32F, ksize=3)
# img_lap_res_norm_bw = cv2.normalize(img_lap_res_bw, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# img_log_res_bw = cv2.Laplacian(img_gauss_res_bw, cv2.CV_32F, ksize=3)
# img_log_res_norm_bw = cv2.normalize(img_log_res_bw, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# img_log_res = cv2.Laplacian(img_gauss_conv, cv2.CV_32F, ksize=3)
# img_log_res_norm_bw = cv2.normalize(img_log_res, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Original Image (B&W)', img_bw)
cv2.imshow('Original Image (Color)', img_bgr)

# cv2.imshow('Blue Channel', b)
# cv2.imshow('Green Channel', g)
# cv2.imshow('Red Channel', r)

cv2.imshow('Gaussian Blurr Convolved Image (b&w)', img_gauss_conv_bw)
cv2.imshow('Normalized Gaussian Blurr Convolved Image (b&w)', img_gauss_norm_bw)

cv2.imshow('Gaussian Blurr Convolved Image (color)', img_gauss_conv)
cv2.imshow('Normalized Gaussian Blurr Convolved Image (color)', img_gauss_norm)

# cv2.imshow('Gaussian Blurred Image', img_gauss_res)
# cv2.imshow('Normalized Gaussian Blurred Image', img_gauss_res_norm)

# cv2.imshow('Difference of Gaussian Convolved Image (b&w)', img_dog_mag_bw)
# cv2.imshow('Normalized Difference of Gaussian Convolved Image (b&w)', img_dog_norm_bw)

# cv2.imshow('4N Laplacian Convoluted Image (b&w)', img_lap4_conv_bw)
# cv2.imshow('4N Normalized Laplacian Convoluted Image (b&w)', img_lap4_norm_bw)

# cv2.imshow('8N Laplacian Convoluted Image (b&w)', img_lap8_conv_bw)
# cv2.imshow('8N Normalized Laplacian Convoluted Image (b&w)', img_lap8_norm_bw)

# cv2.imshow('laplacian image (b&w)', img_lap_res_bw)
# cv2.imshow('Normalized laplacian image (b&w)', img_lap_res_norm_bw)

cv2.imshow('Laplacian of Gaussian Convolved Image (b&w)', img_log_conv_bw)
cv2.imshow('Normalized Laplacian of Gaussian Convolved Image (b&w)', img_log_norm_bw)

# cv2.imshow('Laplacian of Gaussian Image (b&w)', img_log_res_bw)
# cv2.imshow('Normalized Laplacian of Gaussian Image (b&w)', img_log_res_norm_bw)

cv2.imshow('Laplacian of Gaussian Convolved Image (color)', img_log_conv)
cv2.imshow('Normalized Laplacian of Gaussian Convolved Image (color)', img_log_norm)

cv2.waitKey(0)
cv2.destroyAllWindows()
