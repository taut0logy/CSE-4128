#!/usr/bin/env python3
"""
remove_moire_classical.py

Non-ML moire removal pipeline:
- 2D peak detection in FFT
- Gaussian notch filters (for peaks and conjugates)
- Optional radial band-stop where needed
- Iterative refinement
- Spatial postprocessing (median, NLM denoise, CLAHE)
- Wavelet shrinkage on high-frequency bands

Dependencies:
pip install numpy scipy opencv-python matplotlib pywavelets
"""

import sys
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import pywt

# -------------------------
# Utility / filter makers
# -------------------------
def gaussian_notch_reject(shape, u0, v0, sigma):
    """
    Gaussian notch reject centered at (u0, v0) in frequency coords where center is 0.
    shape: (rows, cols)
    u0, v0: coordinates relative to center
    sigma: spread (larger = gentler)
    """
    rows, cols = shape
    U, V = np.meshgrid(np.arange(cols) - cols//2, np.arange(rows) - rows//2)
    D1 = np.sqrt((U - u0)**2 + (V - v0)**2)
    D2 = np.sqrt((U + u0)**2 + (V + v0)**2)
    H = 1.0 - np.exp(-0.5 * (D1**2) / (sigma**2)) * np.exp(-0.5 * (D2**2) / (sigma**2))
    return H

def radial_band_reject(shape, r0, width):
    """
    Gaussian radial band reject around radius r0 (from center).
    """
    rows, cols = shape
    U, V = np.meshgrid(np.arange(cols) - cols//2, np.arange(rows) - rows//2)
    R = np.sqrt(U**2 + V**2)
    H = 1.0 - np.exp(-0.5 * ((R - r0)**2) / (width**2))
    return H

# -------------------------
# 2D peak detection (FFT)
# -------------------------
def detect_fft_peaks(mag, threshold_rel=0.25, min_distance=8, max_peaks=80):
    """
    Detect local peaks in magnitude spectrum away from center.
    Returns list of (row, col) indices.
    threshold_rel: relative threshold of max magnitude to consider
    min_distance: minimum separation (pixels)
    """
    # suppress DC by zeroing a small central region
    r, c = mag.shape
    mag_copy = mag.copy()
    rr, cc = r//2, c//2
    mag_copy[rr-6:rr+7, cc-6:cc+7] = 0

    # local maximum filter
    footprint = np.ones((3,3))
    local_max = ndimage.maximum_filter(mag_copy, footprint=footprint) == mag_copy
    # threshold
    thresh = mag_copy.max() * threshold_rel
    detected = local_max & (mag_copy >= thresh)

    coords = np.column_stack(np.nonzero(detected))
    # sort by magnitude descending
    mags = mag_copy[coords[:,0], coords[:,1]]
    order = np.argsort(-mags)
    coords = coords[order]
    # enforce min distance
    chosen = []
    for (rr0, cc0) in coords:
        if len(chosen) >= max_peaks:
            break
        too_close = False
        for (a,b) in chosen:
            if np.hypot(a-rr0, b-cc0) < min_distance:
                too_close = True
                break
        if not too_close:
            chosen.append((rr0, cc0))
    return chosen

# -------------------------
# Main processing pipeline
# -------------------------
def remove_moire(img, show_debug=False, iter_passes=2):
    """
    img: grayscale uint8 numpy array
    show_debug: if True, show intermediate plots
    iter_passes: how many times to detect & notch (2 is usually enough)
    returns: cleaned image uint8
    """
    rows, cols = img.shape
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)

    H_total = np.ones((rows, cols), dtype=np.float32)

    for it in range(iter_passes):
        # detect peaks in 2D mag
        peaks = detect_fft_peaks(mag, threshold_rel=0.22 - 0.04*it, min_distance=10 - 2*it, max_peaks=60)

        if show_debug:
            print(f"Pass {it+1}: detected {len(peaks)} peaks")

        # convert to centered coords (u,v) relative to center
        for (r0, c0) in peaks:
            u = c0 - cols//2
            v = r0 - rows//2
            # ignore tiny offsets near center
            if np.hypot(u, v) < 6:
                continue
            # sigma scales with radius (wider for farther peaks)
            sigma = max(6, int(0.06 * np.hypot(u, v) + 3))
            Hn = gaussian_notch_reject((rows, cols), u, v, sigma)
            H_total *= Hn

        # also try radial band-stop if there's a strong ring
        # compute radial energy
        U, V = np.meshgrid(np.arange(cols) - cols//2, np.arange(rows) - rows//2)
        R = np.sqrt(U**2 + V**2)
        # build radial profile
        rmax = int(min(rows, cols)//2)
        profile = []
        for rr in range(1, rmax, 2):
            mask = (R >= rr-1) & (R < rr+1)
            profile.append(mag[mask].mean() if np.any(mask) else 0)
        profile = np.array(profile)
        rad_idx = profile.argmax()
        rad_value = profile[rad_idx]
        # if radial peak is significantly high, apply band reject
        if rad_value > profile.mean() + 1.5*profile.std() and rad_idx > 3:
            r0 = rad_idx*2
            width = max(6, int(0.08*r0 + 3))
            Hrad = radial_band_reject((rows, cols), r0, width)
            H_total *= Hrad
            if show_debug:
                print(f"Applied radial band reject at r0={r0}, width={width}")

        # update filtered spectrum and recompute mag for residuals (iterative)
        fshift_filtered = fshift * H_total
        mag = np.log(np.abs(fshift_filtered) + 1)

    # final inverse
    f_ishift = np.fft.ifftshift(fshift * H_total)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # -------------------------
    # Spatial postprocessing
    # -------------------------
    # 1) Median to remove grid spikes
    med = cv2.medianBlur(img_back, 3)

    # 2) Non-local means denoising (keeps texture)
    nlm = cv2.fastNlMeansDenoising(med, h=8, templateWindowSize=7, searchWindowSize=21)

    # 3) CLAHE to restore contrast without overamplifying noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cla = clahe.apply(nlm)

    # 4) Wavelet shrinkage on detail coeffs only (avoid blurring coarse image)
    coeffs = pywt.wavedec2(cla.astype(np.float32), wavelet='db2', level=2)
    cA, details = coeffs[0], coeffs[1:]
    # soft-threshold details (scale threshold to median absolute deviation)
    def thresh_detail(detail_tuple, base_thresh):
        return tuple(pywt.threshold(d, base_thresh, mode='soft') for d in detail_tuple)

    # estimate base threshold from high-frequency subband
    # use robust estimate
    hf = details[-1][0]  # finest horizontal detail
    sigma_est = np.median(np.abs(hf)) / 0.6745 + 1e-9
    base_thresh = max(10.0, sigma_est * 12.0)
    new_details = []
    for d in details:
        new_details.append(thresh_detail(d, base_thresh))
    coeffs_new = [cA] + new_details
    reco = pywt.waverec2(coeffs_new, wavelet='db2')
    reco = np.clip(reco, 0, 255).astype(np.uint8)

    # final gentle median to remove any leftover artifacts
    final = cv2.medianBlur(reco, 3)
    return final, mag, H_total

# -------------------------
# CLI / Example usage
# -------------------------
def main(argv):
    if len(argv) < 2:
        print("Usage: python remove_moire_classical.py input.png [output.png]")
        return
    in_path = argv[1]
    out_path = argv[2] if len(argv) > 2 else "out.png"

    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read", in_path)
        return

    cleaned, final_mag, H = remove_moire(img, show_debug=True, iter_passes=2)
    cv2.imwrite(out_path, cleaned)
    print("Saved cleaned image to", out_path)

    # show comparison
    cv2.imshow("Original", img)
    cv2.imshow("Cleaned", cleaned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
