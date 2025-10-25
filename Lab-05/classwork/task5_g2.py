# Fourier transform - guassian lowpass filter
import cv2
import numpy as np
from matplotlib import pyplot as plt

# take input
img_input = cv2.imread('../../assets/two_noise.jpeg', cv2.IMREAD_GRAYSCALE)
img = img_input.copy()
image_size = img.shape[0] * img.shape[1]


#%%
# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
#ft_shift = ft
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 

D0 = int(input("Enter D0 (radius): "))
n  = int(input("Enter order n: "))

#Apply filter here
def butterworth_notch_reject(img, u0, v0, D0, n=2):
    h,w = img.shape
    M, N = h//2, w//2
    H = np.ones((h,w), dtype=np.float32)
    X, Y = u0 - M, v0 - N
    for i in range(h):
        for j in range(w):
            Dk  = np.sqrt((i-M-X)**2 + (j-N-Y)**2)
            Dk_ = np.sqrt((i-M+X)**2 + (j-N+Y)**2)
            q1 = (D0 / Dk)**(2*n) if Dk != 0 else 1
            q2 = (D0 / Dk_)**(2*n) if Dk_ != 0 else 1
            H[i,j] = 1 / (1 + q1) * 1 / (1 + q2)

    return H

H1 = butterworth_notch_reject(magnitude_spectrum_ac, 272,256, D0, n)
H2 = butterworth_notch_reject(magnitude_spectrum_ac, 261,261, D0, n)

H = H1 * H2

magnitude_spectrum_ac_ = magnitude_spectrum_ac * H
filtered_spectrum = magnitude_spectrum * H

## phase add F(u,v)=∣F(u,v)∣*e^jθ(u,v)
final_result = np.multiply(magnitude_spectrum_ac_, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = cv2.normalize(img_back, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

## plot
cv2.imshow("input", img_input)
cv2.imshow("Butterworth Notch Reject Filter",(H*255).astype(np.uint8))
cv2.imshow("Magnitude Spectrum",magnitude_spectrum)
cv2.imshow("Magnitude Spectrum after filter",cv2.normalize(filtered_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U))
cv2.imshow("Phase", ang_)
cv2.waitKey(0)
cv2.imshow("Inverse transform",img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.figure(figsize=(9,18))

# plt.subplot(5,2,1),plt.imshow(img_input, cmap = 'gray'),plt.title('Input Image'), plt.axis('off')
# plt.subplot(5,2,2),plt.imshow(magnitude_spectrum, cmap = 'gray'),plt.title('Magnitude Spectrum'), plt.axis('off')
# plt.subplot(5,2,3), plt.imshow(ang_, cmap = 'gray'),plt.title('Phase Spectrum'), plt.axis('off')
# plt.subplot(5,2,4),plt.imshow((H*255).astype(np.uint8), cmap = 'gray'),plt.title(f'Butterworth Notch Reject Filter (D0={D0}, n={n})'), plt.axis('off')
# plt.subplot(5,2,5),plt.imshow(cv2.normalize(filtered_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U), cmap = 'gray'),plt.title('Magnitude Spectrum after filter (D0 = 5)'), plt.axis('off')
# plt.subplot(5,2,6),plt.imshow(img_back_scaled, cmap = 'gray'),plt.title('Inverse Transform (D0 = 5)'), plt.axis('off')

# plt.tight_layout()
# plt.show()