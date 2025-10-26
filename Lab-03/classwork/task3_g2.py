import numpy as np
import matplotlib.pyplot as plt
import cv2

def histogram_equalization(img):
    L = 256
    h, w = img.shape
    size = h * w

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    pdf = hist / size

    cdf = np.zeros(256)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]

    trn = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        trn[i] = (cdf[i] * (L-1))

    img_eq = cv2.LUT(img, trn)

    hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256]).flatten()
    pdf_eq = hist_eq / size

    cdf_eq = np.zeros(256)
    cdf_eq[0] = pdf_eq[0]
    for i in range(1, 256):
        cdf_eq[i] = cdf_eq[i-1] + pdf_eq[i]

    return img_eq, hist, pdf, cdf, hist_eq, pdf_eq, cdf_eq

img_color = cv2.imread('../../assets/color_img.jpg', cv2.IMREAD_COLOR)

img_b, img_g, img_r = cv2.split(img_color)

r_eq, hist_r, pdf_r, cdf_r, hist_r_eq, pdf_r_eq, cdf_r_eq = histogram_equalization(img_r)
g_eq, hist_g, pdf_g, cdf_g, hist_g_eq, pdf_g_eq, cdf_g_eq = histogram_equalization(img_g)
b_eq, hist_b, pdf_b, cdf_b, hist_b_eq, pdf_b_eq, cdf_b_eq = histogram_equalization(img_b)

img_rgb_eq = cv2.merge([b_eq, g_eq, r_eq])

plt.figure(figsize=(8, 12))

plt.subplot(3,1,1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(3,1,2)
plt.imshow(cv2.cvtColor(img_rgb_eq, cv2.COLOR_BGR2RGB))
plt.title('RGB Equalized Image')
plt.axis('off')

plt.show()

plt.figure(figsize=(12, 18))

plt.subplot(3,2,1)
plt.bar(range(256), hist_r, width=0.8, color='r')
plt.title('Red Channel Histogram (Original)')
plt.xlim([0, 256])

plt.subplot(3,2,3)
plt.bar(range(256), hist_g, width=0.8, color='g')
plt.title('Green Channel Histogram (Original)')
plt.xlim([0, 256])

plt.subplot(3,2,5)
plt.bar(range(256), hist_b, width=0.8, color='b')
plt.title('Blue Channel Histogram (Original)')
plt.xlim([0, 256])

plt.subplot(3,2,2)
plt.bar(range(256), hist_r_eq, width=0.8, color='r')
plt.title('Red Channel Histogram (Equalized)')
plt.xlim([0, 256])

plt.subplot(3,2,4)
plt.bar(range(256), hist_g_eq, width=0.8, color='g')
plt.title('Green Channel Histogram (Equalized)')
plt.xlim([0, 256])

plt.subplot(3,2,6)
plt.bar(range(256), hist_b_eq, width=0.8, color='b')
plt.title('Blue Channel Histogram (Equalized)')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 18))
plt.subplot(3,2,1)
plt.plot(cdf_r, color='r')
plt.title('Red Channel CDF (Original)')
plt.xlim([0, 256])
plt.ylim([0, 1])

plt.subplot(3,2,3)
plt.plot(cdf_g, color='g')
plt.title('Green Channel CDF (Original)')
plt.xlim([0, 256])
plt.ylim([0, 1])

plt.subplot(3,2,5)
plt.plot(cdf_b, color='b')
plt.title('Blue Channel CDF (Original)')
plt.xlim([0, 256])
plt.ylim([0, 1])

plt.subplot(3,2,2)
plt.plot(cdf_r_eq, color='r')
plt.title('Red Channel CDF (Equalized)')
plt.xlim([0, 256])
plt.ylim([0, 1])

plt.subplot(3,2,4)
plt.plot(cdf_g_eq, color='g')
plt.title('Green Channel CDF (Equalized)')
plt.xlim([0, 256])
plt.ylim([0, 1])

plt.subplot(3,2,6)
plt.plot(cdf_b_eq, color='b')
plt.title('Blue Channel CDF (Equalized)')
plt.xlim([0, 256])
plt.ylim([0, 1])

plt.tight_layout()
plt.show()