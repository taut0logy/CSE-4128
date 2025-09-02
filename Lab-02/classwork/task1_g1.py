import cv2
import numpy as np

def gaussian(u, v, sigma = 1):
    g = np.exp(-(u**2 + v**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return g

def gaussian_derivative_x(u, v, sigma):
    g_x = -u * gaussian(u, v, sigma) / (sigma**2)
    return g_x

def gaussian_derivative_y(u, v, sigma):
    g_y = -v * gaussian(u, v, sigma) / (sigma**2)
    return g_y

def gaussian_derivative_kernels(m=7, sigma=1):
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    kernel_x = np.zeros((m, m), dtype=np.float32)
    kernel_y = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            kernel_x[i, j] = gaussian_derivative_x(i - k, j - k, sigma)
            kernel_y[i, j] = gaussian_derivative_y(i - k, j - k, sigma)
    return kernel_x, kernel_y

def double_threshold(image, T_min=64, T_max=128):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < T_min:
                image[i, j] = 0
            elif image[i, j] < T_max:
                image[i, j] = 127
            else:
                image[i, j] = 255

img = cv2.imread('../../../assets/Lena.jpg', cv2.IMREAD_GRAYSCALE)

kernel_x, kernel_y = gaussian_derivative_kernels()

img_gx = cv2.filter2D(img, cv2.CV_32F, kernel=kernel_x)
img_gy = cv2.filter2D(img, cv2.CV_32F, kernel=kernel_y)

img_gx_norm = cv2.normalize(img_gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img_gy_norm = cv2.normalize(img_gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

grad_mag = cv2.magnitude(img_gx.astype(np.float32), img_gy.astype(np.float32))
grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img_thresh = grad_mag_norm.copy()
double_threshold(img_thresh)

print("KERNEL (X direction):")
print(kernel_x)

print("\nKERNEL (Y direction):")
print(kernel_y)

cv2.imshow('Original Grayscale Image', img)
cv2.imshow('Gradient X', img_gx_norm)
cv2.imshow('Gradient Y', img_gy_norm)
cv2.imshow('Gradient Magnitude', grad_mag_norm)
cv2.imshow('Double Threshold', img_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()