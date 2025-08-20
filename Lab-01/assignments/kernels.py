import numpy as np

def gaussian(u, v, sigma = 1):
    g = np.exp(-(u**2 + v**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return g

def gaussian2(u, v, sigma1, sigma2):
    g = np.exp(-0.5 * ((u**2 / sigma1**2) + (v**2 / sigma2**2))) / (2 * np.pi * sigma1 * sigma2)
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

def gaussian_derivative_kernel_first(m, sigma):
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    kernel_x = np.zeros((m, m), dtype=np.float32)
    kernel_y = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            g = gaussian(i - k, j - k, sigma)
            kernel_x[i, j] = - (i - k) * g / sigma**2
            kernel_y[i, j] = - (j - k) * g / sigma**2

    return kernel_x, kernel_y

def laplacian_of_gaussian_kernel(m, sigma):
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    laplacian = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            laplacian[i, j] = laplacian_of_gaussian(i - k, j - k, sigma)
    

    return laplacian