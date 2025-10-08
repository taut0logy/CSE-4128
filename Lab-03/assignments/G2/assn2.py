import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img = cv2.imread('../../../assets/histogram.jpg', cv2.IMREAD_GRAYSCALE)

def get_hist_pdf_cdf(img):
    L = 256
    h, w = img.shape
    size = h * w
    hist = cv2.calcHist([img], [0], None, [L], [0, L])
    hist = cv2.transpose(hist)[0]
    pdf = hist / size
    cdf = np.zeros(L)
    cdf[0] = pdf[0]
    for i in range(1, L):
        cdf[i] = cdf[i-1] + pdf[i]
    return hist, pdf, cdf

def gaussian_pdf(x, mu, sigma):
    assert sigma > 0, "sigma must be positive"
    c = 1 / (sigma * np.sqrt(2 * np.pi))
    e = -0.5 * ((x - mu) / sigma) ** 2
    pdf = c * np.exp(e)
    if(math.isnan(pdf) or math.isinf(pdf)):
        pdf = 0
    return pdf

def target_histogram(mu1, sigma1, mu2, sigma2):
    assert mu1 > 0 and sigma1 > 0 and mu2 > 0 and sigma2 > 0, "mu must be positive float and sigma must be positive float"
    L = 256
    hist = np.zeros(L, dtype=np.float64)
    for x in range(L):
        hist[x] = gaussian_pdf(x, mu1, sigma1) + gaussian_pdf(x, mu2, sigma2)
        
    hist = hist / np.sum(hist)
    hist = hist * (img.shape[0] * img.shape[1])
    pdf = hist / np.sum(hist)
    cdf = np.zeros(L)
    cdf[0] = pdf[0]
    for i in range(1, L):
        cdf[i] = cdf[i-1] + pdf[i]
    return hist, pdf, cdf

def histogram_matching(source, target_cdf):
    L = 256
    _, _, src_cdf = get_hist_pdf_cdf(source)
    src_cdf_t = np.round(src_cdf * (L - 1)).astype(np.uint8)
    target_cdf_t = np.round(target_cdf * (L - 1)).astype(np.uint8)
    mapping = np.zeros(L, dtype=np.uint8)
    for i in range(L):
        src = src_cdf_t[i]
        diff = np.abs(target_cdf_t - src)
        mapping[i] = np.argmin(diff)
    matched = cv2.LUT(source, mapping)
    return matched, mapping

mu1 = int(input("Enter 1st Gaussian mean parameter (mu, int > 0): "))
sigma1 = float(input("Enter 1st Gaussian standard deviation parameter (sigma, float > 0): "))
mu2 = int(input("Enter 2nd Gaussian mean parameter (mu, int > 0): "))
sigma2 = float(input("Enter 2nd Gaussian standard deviation parameter (sigma, float > 0): "))

L = 256
x = np.arange(L)

target_hist, target_pdf, target_cdf = target_histogram(mu1, sigma1, mu2, sigma2)
# target_hist, target_pdf, target_cdf = target_histogram(30, 8, 165, 20)

img_matched, mapping = histogram_matching(img, target_cdf)

hist_in, pdf_in, cdf_in = get_hist_pdf_cdf(img)
hist_out, pdf_out, cdf_out = get_hist_pdf_cdf(img_matched)

plt.figure(figsize=(6,4))
plt.title("Generated Target Double Gaussian Histogram (PDF)")
plt.bar(x, target_pdf, color='blue')
plt.xlabel("Intensity")
plt.ylabel("Probability")
plt.xlim([0,255])
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(img_matched, cmap='gray')
plt.title("Histogram Matched Image")
plt.axis('off')

plt.subplot(1,3,3)
plt.bar(x, target_pdf, color='blue')
plt.title("Target Double Gaussian PDF")
plt.xlabel("Intensity")
plt.ylabel("Probability")

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))

plt.subplot(1,3,1)
plt.title("Input PDF")
plt.bar(x, pdf_in, color='red', alpha=0.7)
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Probability Density")

plt.subplot(1,3,2)
plt.title("Matched PDF")
plt.bar(x, pdf_out, color='green', alpha=0.7)
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Probability Density")

plt.subplot(1,3,3)
plt.title("Target Double Gaussian PDF")
plt.bar(x, target_pdf, color='blue', alpha=0.7)
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Probability Density")

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))

plt.subplot(1,3,1)
plt.title("Input CDF")
plt.plot(x, cdf_in, color='red')
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Cumulative Density")

plt.subplot(1,3,2)
plt.title("Matched CDF")
plt.plot(x, cdf_out, color='green')
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Cumulative Density")

plt.subplot(1,3,3)
plt.title("Target Double Gaussian CDF")
plt.plot(x, target_cdf, color='blue')
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Cumulative Density")

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))

plt.subplot(1,3,1)
plt.title("Input Histogram (Counts)")
plt.bar(x, hist_in, color='red', alpha=0.7)
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Counts")

plt.subplot(1,3,2)
plt.title("Matched Histogram (Counts)")
plt.bar(x, hist_out, color='green', alpha=0.7)
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Counts")

plt.subplot(1,3,3)
plt.title("Target Histogram (Counts)")
plt.bar(x, target_hist, color='blue', alpha=0.7)
plt.xlim([0,255])
plt.xlabel("Intensity")
plt.ylabel("Counts")

plt.tight_layout()
plt.show()
