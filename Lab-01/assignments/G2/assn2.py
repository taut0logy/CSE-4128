import numpy as np
import cv2
import time

def convolution(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape
    
    assert k_h % 2 == 1 and k_w % 2 == 1, "Kernel size must be odd"
    
    pad_y = (k_h - 1) // 2
    pad_x = (k_w - 1) // 2

    img_bordered = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT)
    img_conv = np.zeros(img_bordered.shape, dtype=np.float32)

    op_count = 0
    stime = time.time()

    for i in range(h):
        for j in range(w):
            sum = 0.0
            for m in range(-pad_y, pad_y + 1):
                for n in range(-pad_x, pad_x + 1):
                    op_count += 1
                    sum += img_bordered[i - m, j - n] * kernel[m + pad_y, n + pad_x]
                    op_count += 1
            img_conv[i, j] = sum
            op_count += 1

    etime = time.time()
    img_norm = np.round(cv2.normalize(img_conv, None, 0, 255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
    img_res = img_norm[pad_y: h + pad_y, pad_x: w + pad_x]

    return img_res, op_count, (etime - stime)

def svd_rank1(kernel):
    U, S, Vt = np.linalg.svd(kernel)
    sigma = S[0]
    u = U[:, [0]]
    v = Vt[[0], :]
    kY = (u * np.sqrt(sigma)).astype(np.float32)
    kX = (v * np.sqrt(sigma)).astype(np.float32)
    K1 = np.outer(kY.flatten(), kX.flatten())
    return kX, kY, K1, float(sigma)

kernel = np.array([
    [0,   1,   2,   1,   0],
    [1,   3,   5,   3,   1], 
    [2,   5,   9,   5,   2],
    [1,   3,   5,   3,   1],
    [0,   1,   2,   1,   0]], dtype=np.float32)

img = cv2.imread('../../../assets/Lena.jpg', cv2.IMREAD_GRAYSCALE)

kX, kY, K1, sigma = svd_rank1(kernel)

decomposition_error = np.abs(kernel - K1)
absolute_error = np.sum(decomposition_error)
relative_error = np.linalg.norm(decomposition_error) / np.linalg.norm(kernel)

print("="*60)
print("KERNEL DECOMPOSITION ANALYSIS")
print("="*60)

print("\nOriginal Kernel (Kernel_1):")
print(kernel)

print(f"\nSingular Value: {sigma:.6f}")

print("\nX Vector Kernel (kX):")
print(kX)

print("\nY Vector Kernel (kY):")
print(kY)

print("\nApproximated Kernel (Kernel_approx = outer product of kY and kX):")
print(K1)

print("\nAbsolute Decomposition Error (|Kernel_1 - Kernel_approx|):")
print(decomposition_error)

print("\nError Statistics:")
print(f"Total Absolute Error: {absolute_error:.6f}")
print(f"Relative Error (Frobenius norm): {relative_error:.6f}")
print(f"Maximum Error: {np.max(decomposition_error):.6f}")
print(f"Mean Error: {np.mean(decomposition_error):.6f}")

# Perform convolutions
img_res, op_count, exec_time = convolution(img, kernel)
img_res_kx, op_count_kx, exec_time_kx = convolution(img, kX)
img_res_ky, op_count_ky, exec_time_ky = convolution(img, kY)
img_res_k1, op_count_k1, exec_time_k1 = convolution(img, K1)

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"Direct Convolution: Operations = {op_count}, Time = {exec_time:.6f} seconds")
print(f"Rank-1 Approximation (KX): Operations = {op_count_kx}, Time = {exec_time_kx:.6f} seconds")
print(f"Rank-1 Approximation (KY): Operations = {op_count_ky}, Time = {exec_time_ky:.6f} seconds")
print(f"Rank-1 Approximation (K1): Operations = {op_count_k1}, Time = {exec_time_k1:.6f} seconds")

# Display images
cv2.imshow('Original Image', img)
cv2.imshow('Convolved Image (Direct)', img_res)
cv2.imshow('Convolved Image (KX)', img_res_kx)
cv2.imshow('Convolved Image (KY)', img_res_ky)
cv2.imshow('Convolved Image (K1 - Approximated)', img_res_k1)

cv2.waitKey(0)
cv2.destroyAllWindows()