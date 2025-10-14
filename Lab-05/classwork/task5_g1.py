# Fourier transform - guassian lowpass filter
import cv2
import numpy as np
from matplotlib import pyplot as plt

# take input
img_input = cv2.imread('../../assets/pnois2.jpg', cv2.IMREAD_GRAYSCALE)
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

def filter(img, x, y, d):
    h,w=img.shape
    M,N=h // 2, w // 2
    X, Y = x-M, y-N
    flt=np.ones_like(img)
    for i in range(h):
        for j in range(w):
            d1 = np.sqrt((i-M-X)**2+(j-N-Y)**2)
            d2 = np.sqrt((i-M+X)**2+(j-N+Y)**2)
            if(d1<=d or d2<=d):
                flt[i,j]=0

    return flt

d=int(input("Enter d: "))

#Apply filter here

notch = filter(magnitude_spectrum_ac, 261, 261, d)
magnitude_spectrum_ac = magnitude_spectrum_ac * notch
filtered_spectrum = magnitude_spectrum * notch

## phase add F(u,v)=∣F(u,v)∣*e^jθ(u,v)
final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = cv2.normalize(img_back, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)


## plot
cv2.imshow("input", img_input)
cv2.imshow("Notch Filter", notch*255)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum)
cv2.imshow("Magnitude Spectrum after filter",cv2.normalize(filtered_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U))
cv2.imshow("Phase", ang_)
cv2.waitKey(0)
cv2.imshow("Inverse transform",img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()