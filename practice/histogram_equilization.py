import cv2
import numpy as np
import matplotlib.pylab as plt

def histogram_equilization(image):
    h,w = image.shape
    size = h*w
    L=256

    histogram = np.zeros(L, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            histogram[image[i,j]] += 1

    prob = histogram / size

    cum_sum_prob = np.zeros(L, dtype=np.float32)
    cum_sum_prob[0] = prob[0]
    trans_arr = np.zeros_like(cum_sum_prob, dtype=np.uint8)
    for i in range(1, L):
        cum_sum_prob[i] = cum_sum_prob[i-1] + prob[i]
        trans_arr[i] = np.round((L-1) * cum_sum_prob[i]).astype(np.uint8)

    img_eq = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            img_eq[i,j] = trans_arr[image[i,j]]
            
    histogram_eq = np.zeros(L, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            histogram_eq[img_eq[i,j]] += 1

    return histogram, prob, cum_sum_prob, img_eq, histogram_eq

img = cv2.imread("assets/Lena.jpg", cv2.IMREAD_GRAYSCALE)

histogram, prob, cum_sum_prob, img_eq, histogram_eq = histogram_equilization(img)


plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.axis("off")
plt.imshow(img, cmap="gray")

plt.subplot(2, 3, 2)
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.bar(range(256), histogram, color='green', alpha=0.7)

plt.subplot(2, 3, 3)
plt.title("Probability Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability")
plt.grid(True, alpha=0.3)
plt.plot(range(256), prob, color='blue')

plt.subplot(2, 3, 4)
plt.title("Cumulative Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Probability")
plt.grid(True, alpha=0.3)
plt.plot(range(256), cum_sum_prob, color='black')

plt.subplot(2, 3, 5)
plt.title("Equalized Image")
plt.axis("off")
plt.imshow(img_eq, cmap="gray")

plt.subplot(2, 3, 6)
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.bar(range(256), histogram_eq, color='black', alpha=0.7)

plt.tight_layout()
plt.show()