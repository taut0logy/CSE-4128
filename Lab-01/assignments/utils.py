import cv2
import numpy as np

def convolve(img, kernel):
    h, w = img.shape
    k_h, k_w = kernel.shape
    
    assert k_h == k_w, "Kernel must be square"
    assert k_h % 2 == 1 and k_w % 2 == 1, "Kernel size must be odd"
    
    pad = (k_h - 1) // 2
    
    img_bordered = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    img_conv = np.zeros(img_bordered.shape, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            sum = 0.0
            for m in range(-pad, pad + 1):
                for n in range(-pad, pad + 1):
                    sum += img_bordered[i - m, j - n] * kernel[m + pad, n + pad]
            img_conv[i, j] = sum
            
    img_norm = np.round(cv2.normalize(img_conv, None, 0, 255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
    img_res = img_norm[pad: h + pad, pad: w + pad]
    
    return img_bordered, img_conv, img_norm, img_res

def mean_filter(img, size=3):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return convolve(img, kernel)
