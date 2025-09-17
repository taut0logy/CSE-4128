import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def apply_fft(image):
    """Apply 2D Fast Fourier Transform and shift zero frequency to center"""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift

def apply_ifft(fshift):
    """Apply inverse 2D Fast Fourier Transform"""
    f_ishift = np.fft.ifftshift(fshift)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)
    return image_back

def create_butterworth_notch_filter(shape, centers, d0=30, n=2):
    """Create a Butterworth notch rejection filter"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    filter = np.ones((rows, cols))
    
    for center in centers:
        u0, v0 = center
        # Create distance matrices
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        u, v = np.meshgrid(u, v, indexing='ij')
        
        # Distance from the center of the notch
        d1 = np.sqrt((u - (u0 - crow))**2 + (v - (v0 - ccol))**2)
        d2 = np.sqrt((u + (u0 - crow))**2 + (v + (v0 - ccol))**2)
        
        # Butterworth notch filter
        filter *= 1 / (1 + (d0**2 / (d1 * d2 + 1e-10))**n)
    
    return filter

def create_gaussian_notch_filter(shape, centers, d0=30):
    """Create a Gaussian notch rejection filter"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    filter = np.ones((rows, cols))
    
    for center in centers:
        u0, v0 = center
        # Create distance matrices
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        u, v = np.meshgrid(u, v, indexing='ij')
        
        # Distance from the center of the notch
        d1 = np.sqrt((u - (u0 - crow))**2 + (v - (v0 - ccol))**2)
        d2 = np.sqrt((u + (u0 - crow))**2 + (v + (v0 - ccol))**2)
        
        # Gaussian notch filter
        filter *= (1 - np.exp(-d1**2/(2*d0**2))) * (1 - np.exp(-d2**2/(2*d0**2)))
    
    return filter

def create_ideal_notch_filter(shape, centers, d0=30):
    """Create an ideal notch rejection filter"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    filter = np.ones((rows, cols))
    
    for center in centers:
        u0, v0 = center
        # Create distance matrices
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        u, v = np.meshgrid(u, v, indexing='ij')
        
        # Distance from the center of the notch
        d1 = np.sqrt((u - (u0 - crow))**2 + (v - (v0 - ccol))**2)
        d2 = np.sqrt((u + (u0 - crow))**2 + (v + (v0 - ccol))**2)
        
        # Ideal notch filter
        filter[(d1 <= d0) | (d2 <= d0)] = 0
    
    return filter

def detect_peaks(magnitude_spectrum, threshold_ratio=0.8, min_distance=10):
    """Detect peaks in the magnitude spectrum"""
    # Apply threshold to find bright spots
    threshold = threshold_ratio * np.max(magnitude_spectrum)
    mask = magnitude_spectrum > threshold
    
    # Find coordinates of peaks
    y_peaks, x_peaks = np.where(mask)
    
    # Group nearby peaks and find the strongest one in each group
    peaks = []
    for y, x in zip(y_peaks, x_peaks):
        # Check if this peak is far enough from existing peaks
        if all(np.sqrt((y-py)**2 + (x-px)**2) > min_distance for py, px in peaks):
            peaks.append((y, x))
    
    return peaks

def remove_moire_pattern(image_path, output_path, filter_type='butterworth', d0=30, n=2, 
                         threshold_ratio=0.8, min_distance=10, manual_peaks=None,
                         apply_hist_eq=False, apply_median=False, kernel_size=5,
                         show_plots=False, save_plots=False):
    """
    Remove moiré patterns from an image using frequency domain filtering
    
    Parameters:
    - image_path: Path to the input image
    - output_path: Path to save the output image
    - filter_type: Type of filter to use ('butterworth', 'gaussian', 'ideal')
    - d0: Cutoff frequency for the filter
    - n: Order of the Butterworth filter
    - threshold_ratio: Ratio for peak detection threshold
    - min_distance: Minimum distance between peaks
    - manual_peaks: List of manual peak coordinates [(y1, x1), (y2, x2), ...]
    - apply_hist_eq: Apply histogram equalization as post-processing
    - apply_median: Apply median filtering as post-processing
    - kernel_size: Kernel size for median filtering
    - show_plots: Show intermediate plots
    - save_plots: Save intermediate plots
    """
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize intensity values
    gray = gray.astype(np.float32) / 255.0
    
    # Apply FFT
    fshift = apply_fft(gray)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Detect peaks in the magnitude spectrum
    if manual_peaks is None:
        peaks = detect_peaks(magnitude_spectrum, threshold_ratio, min_distance)
    else:
        peaks = manual_peaks
    
    # Create notch filter based on the selected type
    if filter_type == 'butterworth':
        notch_filter = create_butterworth_notch_filter(gray.shape, peaks, d0, n)
    elif filter_type == 'gaussian':
        notch_filter = create_gaussian_notch_filter(gray.shape, peaks, d0)
    elif filter_type == 'ideal':
        notch_filter = create_ideal_notch_filter(gray.shape, peaks, d0)
    else:
        raise ValueError("Filter type must be 'butterworth', 'gaussian', or 'ideal'")
    
    # Apply the filter
    filtered_fshift = fshift * notch_filter
    
    # Apply inverse FFT
    filtered_image = apply_ifft(filtered_fshift)
    
    # Post-processing
    if apply_hist_eq:
        filtered_image = cv2.equalizeHist((filtered_image * 255).astype(np.uint8))
        filtered_image = filtered_image.astype(np.float32) / 255.0
    
    if apply_median:
        filtered_image = cv2.medianBlur((filtered_image * 255).astype(np.uint8), kernel_size)
        filtered_image = filtered_image.astype(np.float32) / 255.0
    
    # Convert back to 8-bit
    filtered_image = (filtered_image * 255).astype(np.uint8)
    
    # Save the result
    cv2.imwrite(output_path, filtered_image)
    
    # Generate plots if requested
    if show_plots or save_plots:
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Magnitude spectrum
        plt.subplot(2, 3, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.axis('off')
        
        # Mark peaks on magnitude spectrum
        plt.subplot(2, 3, 3)
        plt.imshow(magnitude_spectrum, cmap='gray')
        for peak in peaks:
            plt.plot(peak[1], peak[0], 'ro')
        plt.title('Detected Peaks')
        plt.axis('off')
        
        # Filter
        plt.subplot(2, 3, 4)
        plt.imshow(notch_filter, cmap='gray')
        plt.title('Notch Filter')
        plt.axis('off')
        
        # Filtered magnitude spectrum
        filtered_magnitude = 20 * np.log(np.abs(filtered_fshift) + 1)
        plt.subplot(2, 3, 5)
        plt.imshow(filtered_magnitude, cmap='gray')
        plt.title('Filtered Spectrum')
        plt.axis('off')
        
        # Restored image
        plt.subplot(2, 3, 6)
        plt.imshow(filtered_image, cmap='gray')
        plt.title('Restored Image')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = output_path.replace('.png', '_analysis.png')
            plt.savefig(plot_path)
            print(f"Analysis plot saved to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return filtered_image, peaks

def main():
    parser = argparse.ArgumentParser(description='Remove moiré patterns from images using frequency domain filtering')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--filter-type', choices=['butterworth', 'gaussian', 'ideal'], 
                        default='butterworth', help='Type of filter to use')
    parser.add_argument('--d0', type=float, default=30, help='Cutoff frequency for the filter')
    parser.add_argument('--n', type=int, default=2, help='Order of the Butterworth filter')
    parser.add_argument('--threshold-ratio', type=float, default=0.8, 
                        help='Ratio for peak detection threshold (0-1)')
    parser.add_argument('--min-distance', type=int, default=10, 
                        help='Minimum distance between peaks')
    parser.add_argument('--manual-peaks', nargs='+', type=int, 
                        help='Manual peak coordinates as y1 x1 y2 x2 ...')
    parser.add_argument('--hist-eq', action='store_true', 
                        help='Apply histogram equalization as post-processing')
    parser.add_argument('--median', action='store_true', 
                        help='Apply median filtering as post-processing')
    parser.add_argument('--kernel-size', type=int, default=5, 
                        help='Kernel size for median filtering')
    parser.add_argument('--show-plots', action='store_true', 
                        help='Show intermediate plots')
    parser.add_argument('--save-plots', action='store_true', 
                        help='Save intermediate plots')
    
    args = parser.parse_args()
    
    # Process manual peaks if provided
    manual_peaks = None
    if args.manual_peaks:
        if len(args.manual_peaks) % 2 != 0:
            raise ValueError("Manual peaks must be provided as pairs of coordinates")
        manual_peaks = [(args.manual_peaks[i], args.manual_peaks[i+1]) 
                        for i in range(0, len(args.manual_peaks), 2)]
    
    # Run the moiré removal
    try:
        filtered_image, peaks = remove_moire_pattern(
            args.input, args.output, args.filter_type, args.d0, args.n,
            args.threshold_ratio, args.min_distance, manual_peaks,
            args.hist_eq, args.median, args.kernel_size,
            args.show_plots, args.save_plots
        )
        print(f"Successfully processed image. Detected {len(peaks)} peaks.")
        print(f"Output saved to {args.output}")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()