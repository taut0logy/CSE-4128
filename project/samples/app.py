import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pywt

# ------------------------------
# Gaussian Notch Filter Function
# ------------------------------
def gaussian_notch_filter(shape, cutoff, order, u_k, v_k):
    """Create a Gaussian notch reject filter."""
    P, Q = shape
    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    U = U - Q // 2
    V = V - P // 2
    Dk = np.sqrt((U - u_k)**2 + (V - v_k)**2)
    Dk_ = np.sqrt((U + u_k)**2 + (V + v_k)**2)
    H = 1 - np.exp(-0.5 * ((Dk / cutoff)**(2*order))) * np.exp(-0.5 * ((Dk_ / cutoff)**(2*order)))
    return H

# ------------------------------
# Load Image
# ------------------------------
image_path = "sample.png"  # your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

# ------------------------------
# FFT
# ------------------------------
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

# ------------------------------
# Detect Peaks in FFT Spectrum
# ------------------------------
proj_x = magnitude_spectrum[rows//2, :]  # horizontal slice through center
peaks_x, _ = find_peaks(proj_x, distance=20, height=np.max(proj_x)*0.4)

proj_y = magnitude_spectrum[:, cols//2]  # vertical slice through center
peaks_y, _ = find_peaks(proj_y, distance=20, height=np.max(proj_y)*0.4)

# ------------------------------
# Build Automatic Notch Filter
# ------------------------------
H = np.ones((rows, cols))
for px in peaks_x:
    H *= gaussian_notch_filter((rows, cols), cutoff=12, order=2, u_k=px-cols//2, v_k=0)
for py in peaks_y:
    H *= gaussian_notch_filter((rows, cols), cutoff=12, order=2, u_k=0, v_k=py-rows//2)

# ------------------------------
# Apply Filter in Frequency Domain
# ------------------------------
fshift_filtered = fshift * H
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Normalize
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ------------------------------
# Wavelet Denoising
# ------------------------------
coeffs = pywt.wavedec2(img_back, 'db1', level=2)
coeffs_thresh = [
    pywt.threshold(c, 15, mode='soft') if isinstance(c, np.ndarray) else c
    for c in coeffs
]
img_denoised = pywt.waverec2(coeffs_thresh, 'db1')

# Normalize final result
img_final = cv2.normalize(img_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ------------------------------
# Show Results
# ------------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title("FFT Spectrum (Peaks)")
plt.subplot(1,3,3), plt.imshow(img_final, cmap='gray'), plt.title("Enhanced Moire Removed")
plt.show()
