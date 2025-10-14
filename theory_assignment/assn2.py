import cv2
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------
# ---- starter image ----

arr=[[61, 65, 64, 67, 60, 60, 58, 57, 54, 54, 54, 55, 52],
    [61, 63, 63, 65, 64, 177, 58, 55, 57, 54, 53, 56, 53],
    [63, 66, 64, 67, 175, 178, 179, 57, 53, 56, 55, 55, 54],
    [60, 65, 62, 177, 176, 176, 182, 180, 178, 54, 55, 53, 55],
    [23, 20, 23, 177, 177, 175, 178, 176, 180, 54, 53, 53, 56],
    [20, 22, 23, 178, 176, 175, 178, 179, 179, 56, 56, 53, 55],
    [23, 20, 23, 177, 175, 175, 176, 177, 180, 52, 53, 55, 54],
    [22, 21, 23, 176, 175, 177, 177, 178, 178, 54, 54, 54, 54],
    [21, 22, 25, 176, 179, 177, 181, 175, 179, 57, 54, 56, 54],
    [21, 23, 21, 76, 78, 78, 76, 81, 79, 54, 56, 56, 55],
    [21, 21, 80, 77, 75, 75, 79, 80, 77, 52, 54, 55, 55],
    [21, 80, 75, 76, 79, 76, 75, 77, 76, 55, 53, 54, 57],
    [20, 80, 77, 76, 79, 78, 80, 78, 79, 53, 54, 55, 57]]

img = np.array(arr, dtype=np.float32)
# ---------------------------

# Output filename
OUT_TXT = "output.txt"

# Utility: pretty print matrix with tabulate and also write to file
def write_section(f, title, matrix, fmt="d", precision=None):
    header = f"\n=== {title} ===\n"
    print(header)
    f.write(header)
    # convert to list-of-lists
    if precision is not None:
        # Show floating point values with specified precision
        mat_disp = [[round(float(x), precision) for x in row] for row in matrix.tolist()]
    else:
        # Default: try to convert to integers (for floored values)
        try:
            mat_disp = matrix.astype(int).tolist()
        except Exception:
            # fallback: convert each element to python number
            mat_disp = [[float(x) for x in row] for row in matrix.tolist()]
    table = tabulate(mat_disp, tablefmt="plain")
    print(table)
    f.write(table + "\n")

def write_scalar(f, title, value):
    line = f"{title}: {value}\n"
    print(line)
    f.write(line)

# Gaussian helper functions (based on your starter functions)
def gaussian(u, v, sigma=1.0):
    return np.exp(-(u**2 + v**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

def gaussian_blurr_kernel(m, sigma):
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    kernel = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            kernel[i, j] = gaussian(i - k, j - k, sigma)
    kernel = kernel / np.sum(kernel)  # normalize so sum equals 1
    return kernel

def gaussian_derivative_kernel_first(m, sigma):
    assert m % 2 == 1, "Kernel size must be odd"
    k = (m - 1) // 2
    kernel_x = np.zeros((m, m), dtype=np.float32)
    kernel_y = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            g = gaussian(i - k, j - k, sigma)
            # Derivative of Gaussian: -x*G(x,y)/sigma^2 for x direction
            kernel_x[i, j] = - (j - k) * g / (sigma**2)  # Note: j is x direction (columns)
            kernel_y[i, j] = - (i - k) * g / (sigma**2)  # Note: i is y direction (rows)
    # Ensure derivative property (sum should be approximately 0)
    kernel_x -= np.mean(kernel_x)
    kernel_y -= np.mean(kernel_y)
    # Normalize so that sum of absolute values equals 1 (assignment requirement)
    sum_abs_x = np.sum(np.abs(kernel_x))
    sum_abs_y = np.sum(np.abs(kernel_y))
    if sum_abs_x != 0:
        kernel_x = kernel_x / sum_abs_x
    if sum_abs_y != 0:
        kernel_y = kernel_y / sum_abs_y
    return kernel_x, kernel_y

# Floor (round down) wrapper used as required by assignment
def floor_array(a):
    return np.floor(a).astype(np.float32)

# Non-maximum suppression (3x3)
def non_maximum_suppression_3x3(img_thresh):
    h, w = img_thresh.shape
    out = np.zeros_like(img_thresh)
    # iterate over pixels
    for i in range(h):
        for j in range(w):
            val = img_thresh[i, j]
            if val <= 0:
                continue
            # define 3x3 neighborhood indices
            i0 = max(0, i-1)
            i1 = min(h-1, i+1)
            j0 = max(0, j-1)
            j1 = min(w-1, j+1)
            window = img_thresh[i0:i1+1, j0:j1+1]
            # local maximum
            if val >= np.max(window):
                # if tie, keep only if it is *strict* maximum or equals the max: assignment said retain strongest local responses
                out[i, j] = val
    return out

# ----- Parameters & kernels per assignment -----
deriv_m = 3
deriv_sigma = 0.5
win_m = 3
win_sigma = 0.6
kappa = 0.04

# compute derivative kernels (3x3 sigma=0.5) with abs-sum=1 property
kernel_dx, kernel_dy = gaussian_derivative_kernel_first(deriv_m, deriv_sigma)

# compute gaussian window (3x3 sigma=0.6) normalized to sum=1 (for smoothing M elements)
gaussian_window = gaussian_blurr_kernel(win_m, win_sigma)

# Prepare file for orderly output
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("Harris - Stephen corner detection results\n")
    f.write("Using provided image array (unchanged)\n")
    f.write(f"Image shape: {img.shape}\n")
    
    # write the image
    write_section(f, "Input Image", img)

    # write kernels
    write_section(f, "Derivative Kernel - Kx (3x3, sigma=0.5) (sum(abs)=1)", kernel_dx, precision=6)
    write_section(f, "Derivative Kernel - Ky (3x3, sigma=0.5) (sum(abs)=1)", kernel_dy, precision=6)
    write_section(f, "Gaussian Window (3x3, sigma=0.6) (sum=1)", gaussian_window, precision=6)

    # compute gradients with border replication (assignment: border replication for gradient computation)
    # Use cv2.filter2D - note cv2 expects float32 kernels
    Ix = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel_dx.astype(np.float32), borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel_dy.astype(np.float32), borderType=cv2.BORDER_REPLICATE)

    # floor (round down) intermediate values as required
    Ix = floor_array(Ix)
    Iy = floor_array(Iy)
    write_section(f, "Ix (image derivative in x) - floored", Ix)
    write_section(f, "Iy (image derivative in y) - floored", Iy)

    # compute Ix^2, Iy^2, IxIy (intermediate)
    Ix2 = floor_array(Ix * Ix)
    Iy2 = floor_array(Iy * Iy)
    Ixy = floor_array(Ix * Iy)
    write_section(f, "Ix^2 (before smoothing) - floored", Ix2)
    write_section(f, "Iy^2 (before smoothing) - floored", Iy2)
    write_section(f, "IxIy (before smoothing) - floored", Ixy)

    # Smooth these using gaussian window with zero padding (assignment: zero padding during smoothing)
    # To enforce zero padding with cv2.filter2D, use borderType=cv2.BORDER_CONSTANT (default value 0)
    # Convert gaussian_window to float32
    gk = gaussian_window.astype(np.float32)
    Ix2_smooth = cv2.filter2D(Ix2, ddepth=cv2.CV_32F, kernel=gk, borderType=cv2.BORDER_CONSTANT)
    Iy2_smooth = cv2.filter2D(Iy2, ddepth=cv2.CV_32F, kernel=gk, borderType=cv2.BORDER_CONSTANT)
    Ixy_smooth = cv2.filter2D(Ixy, ddepth=cv2.CV_32F, kernel=gk, borderType=cv2.BORDER_CONSTANT)

    # floor the smoothed arrays
    Ix2_smooth = floor_array(Ix2_smooth)
    Iy2_smooth = floor_array(Iy2_smooth)
    Ixy_smooth = floor_array(Ixy_smooth)

    write_section(f, "Ix^2 smoothed (using 3x3 Gaussian window, zero padding) - floored", Ix2_smooth)
    write_section(f, "Iy^2 smoothed (using 3x3 Gaussian window, zero padding) - floored", Iy2_smooth)
    write_section(f, "IxIy smoothed (using 3x3 Gaussian window, zero padding) - floored", Ixy_smooth)

    # Compute M components per pixel (already have them: Ix2_smooth, Iy2_smooth, Ixy_smooth)
    # Compute det(M) and trace(M)^2
    detM = floor_array(Ix2_smooth * Iy2_smooth - (Ixy_smooth ** 2))
    traceM = floor_array(Ix2_smooth + Iy2_smooth)
    traceM_sq = floor_array(traceM ** 2)

    write_section(f, "det(M) - floored", detM)
    write_section(f, "trace(M)^2 - floored", traceM_sq)

    # Compute cornerness response R = det(M) - k * trace(M)^2
    # According to instruction, intermediate values are floored. We'll compute R and floor it.
    R_raw = detM - (kappa * traceM_sq)
    # Floor R as intermediate per instruction
    R_raw = np.floor(R_raw)

    write_section(f, "R (raw) = det(M) - k * trace(M)^2 (floored)", R_raw)

    # Scale response map to 0-255 (explicitly requested)
    R_min = float(np.min(R_raw))
    R_max = float(np.max(R_raw))
    write_scalar(f, "R min (raw)", R_min)
    write_scalar(f, "R max (raw)", R_max)

    # If all values are same, scaling will produce zeros; handle that
    if R_max == R_min:
        R_scaled = np.zeros_like(R_raw)
    else:
        # scale linearly to [0, 255]
        R_scaled = (R_raw - R_min) / (R_max - R_min) * 255.0
    # floor scaled map also (the assignment didn't explicitly say to floor scaled map but said round down all intermediate values)
    R_scaled = np.floor(R_scaled)

    write_section(f, "R scaled (0-255) - floored", R_scaled)

    # Thresholding: T = mean(R) + 0.7 * std(R). Use mean/std of the raw (floored) R values
    R_values = R_raw.flatten()
    mean_R = float(np.mean(R_values))
    std_R = float(np.std(R_values, ddof=0))
    T = mean_R + 0.7 * std_R

    write_scalar(f, "mean(R) (used for threshold)", mean_R)
    write_scalar(f, "std(R) (used for threshold)", std_R)
    write_scalar(f, "Threshold T = mean(R) + 0.7*std(R)", T)

    # Apply threshold-to-zero
    R_thresh = np.copy(R_raw)
    R_thresh[R_thresh <= T] = 0.0
    # floor again for safety
    R_thresh = np.floor(R_thresh)

    write_section(f, "R after threshold-to-zero (floored)", R_thresh)

    # Non-maximum suppression (3x3) on thresholded image
    R_nms = non_maximum_suppression_3x3(R_thresh)
    # floor again (should be integers already)
    R_nms = np.floor(R_nms)

    write_section(f, "R after 3x3 Non-Maximum Suppression (floored)", R_nms)

    # Summaries
    write_scalar(f, "Number of nonzero responses after thresholding", int(np.count_nonzero(R_thresh)))
    write_scalar(f, "Number of corners after NMS", int(np.count_nonzero(R_nms)))

    # If you want to list corner coordinates (row, col, value)
    corners = []
    h, w = R_nms.shape
    for i in range(h):
        for j in range(w):
            val = R_nms[i, j]
            if val != 0:
                corners.append((int(i), int(j), int(val)))
    if corners:
        header = "\n=== Detected corners (row, col, value) ===\n"
        print(header)
        f.write(header)
        f.write(tabulate(corners, headers=["row", "col", "value"], tablefmt="plain"))
        f.write("\n")
        print(tabulate(corners, headers=["row", "col", "value"], tablefmt="plain"))
    else:
        header = "\n=== Detected corners (row, col, value) ===\nNo corners detected after NMS.\n"
        print(header)
        f.write(header)

print(f"\nAll results saved to {OUT_TXT}")

# =============================================================================
# VISUALIZATION SECTION
# =============================================================================
print("\nGenerating visualizations...")

# Set plot style for clean, professional look
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'

# Create a comprehensive figure with all intermediate results
fig = plt.figure(figsize=(22, 18))
gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)

# Row 1: Input and Derivative Kernels
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(img, cmap='gray', interpolation='nearest')
ax1.set_title('Input Image', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Column', fontsize=9)
ax1.set_ylabel('Row', fontsize=9)
ax1.tick_params(labelsize=7)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=7)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(kernel_dx, cmap='gray', interpolation='nearest')
ax2.set_title('Derivative Kernel Kx\n(3×3, σ=0.5)', fontsize=11, fontweight='bold', pad=10)
for i in range(deriv_m):
    for j in range(deriv_m):
        val = kernel_dx[i,j]
        color = 'white' if abs(val) > 0.2 else 'black'
        ax2.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')
ax2.set_xticks([])
ax2.set_yticks([])
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=7)

ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(kernel_dy, cmap='gray', interpolation='nearest')
ax3.set_title('Derivative Kernel Ky\n(3×3, σ=0.5)', fontsize=11, fontweight='bold', pad=10)
for i in range(deriv_m):
    for j in range(deriv_m):
        val = kernel_dy[i,j]
        color = 'white' if abs(val) > 0.2 else 'black'
        ax3.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')
ax3.set_xticks([])
ax3.set_yticks([])
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.ax.tick_params(labelsize=7)

ax4 = fig.add_subplot(gs[0, 3])
im4 = ax4.imshow(gaussian_window, cmap='gray', interpolation='nearest')
ax4.set_title('Gaussian Window\n(3×3, σ=0.6)', fontsize=11, fontweight='bold', pad=10)
for i in range(win_m):
    for j in range(win_m):
        val = gaussian_window[i,j]
        color = 'white' if val > 0.3 else 'black'
        ax4.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')
ax4.set_xticks([])
ax4.set_yticks([])
cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cbar4.ax.tick_params(labelsize=7)

# Row 2: Gradients and Their Squares
ax5 = fig.add_subplot(gs[1, 0])
im5 = ax5.imshow(Ix, cmap='gray', interpolation='nearest')
ax5.set_title('Ix (floored)', fontsize=12, fontweight='bold', pad=10)
ax5.set_xlabel('Column', fontsize=9)
ax5.set_ylabel('Row', fontsize=9)
ax5.tick_params(labelsize=7)
cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
cbar5.ax.tick_params(labelsize=7)

ax6 = fig.add_subplot(gs[1, 1])
im6 = ax6.imshow(Iy, cmap='gray', interpolation='nearest')
ax6.set_title('Iy (floored)', fontsize=12, fontweight='bold', pad=10)
ax6.set_xlabel('Column', fontsize=9)
ax6.set_ylabel('Row', fontsize=9)
ax6.tick_params(labelsize=7)
cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
cbar6.ax.tick_params(labelsize=7)

ax7 = fig.add_subplot(gs[1, 2])
im7 = ax7.imshow(Ix2, cmap='gray', interpolation='nearest')
ax7.set_title('Ix² (before smoothing)', fontsize=12, fontweight='bold', pad=10)
ax7.set_xlabel('Column', fontsize=9)
ax7.set_ylabel('Row', fontsize=9)
ax7.tick_params(labelsize=7)
cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
cbar7.ax.tick_params(labelsize=7)

ax8 = fig.add_subplot(gs[1, 3])
im8 = ax8.imshow(Iy2, cmap='gray', interpolation='nearest')
ax8.set_title('Iy² (before smoothing)', fontsize=12, fontweight='bold', pad=10)
ax8.set_xlabel('Column', fontsize=9)
ax8.set_ylabel('Row', fontsize=9)
ax8.tick_params(labelsize=7)
cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
cbar8.ax.tick_params(labelsize=7)

# Row 3: Smoothed Components and Structure Tensor
ax9 = fig.add_subplot(gs[2, 0])
im9 = ax9.imshow(Ix2_smooth, cmap='gray', interpolation='nearest')
ax9.set_title('Ix² (smoothed)', fontsize=12, fontweight='bold', pad=10)
ax9.set_xlabel('Column', fontsize=9)
ax9.set_ylabel('Row', fontsize=9)
ax9.tick_params(labelsize=7)
cbar9 = plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
cbar9.ax.tick_params(labelsize=7)

ax10 = fig.add_subplot(gs[2, 1])
im10 = ax10.imshow(Iy2_smooth, cmap='gray', interpolation='nearest')
ax10.set_title('Iy² (smoothed)', fontsize=12, fontweight='bold', pad=10)
ax10.set_xlabel('Column', fontsize=9)
ax10.set_ylabel('Row', fontsize=9)
ax10.tick_params(labelsize=7)
cbar10 = plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)
cbar10.ax.tick_params(labelsize=7)

ax11 = fig.add_subplot(gs[2, 2])
im11 = ax11.imshow(Ixy_smooth, cmap='gray', interpolation='nearest')
ax11.set_title('IxIy (smoothed)', fontsize=12, fontweight='bold', pad=10)
ax11.set_xlabel('Column', fontsize=9)
ax11.set_ylabel('Row', fontsize=9)
ax11.tick_params(labelsize=7)
cbar11 = plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)
cbar11.ax.tick_params(labelsize=7)

ax12 = fig.add_subplot(gs[2, 3])
im12 = ax12.imshow(detM, cmap='gray', interpolation='nearest')
ax12.set_title('det(M)', fontsize=12, fontweight='bold', pad=10)
ax12.set_xlabel('Column', fontsize=9)
ax12.set_ylabel('Row', fontsize=9)
ax12.tick_params(labelsize=7)
cbar12 = plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)
cbar12.ax.tick_params(labelsize=7)

# Row 4: Response Map and Final Results
ax13 = fig.add_subplot(gs[3, 0])
im13 = ax13.imshow(R_raw, cmap='gray', interpolation='nearest')
ax13.set_title(f'R (raw)\nκ={kappa}', fontsize=12, fontweight='bold', pad=10)
ax13.set_xlabel('Column', fontsize=9)
ax13.set_ylabel('Row', fontsize=9)
ax13.tick_params(labelsize=7)
cbar13 = plt.colorbar(im13, ax=ax13, fraction=0.046, pad=0.04)
cbar13.ax.tick_params(labelsize=7)

ax14 = fig.add_subplot(gs[3, 1])
im14 = ax14.imshow(R_scaled, cmap='gray', interpolation='nearest')
ax14.set_title('R (scaled 0-255)', fontsize=12, fontweight='bold', pad=10)
ax14.set_xlabel('Column', fontsize=9)
ax14.set_ylabel('Row', fontsize=9)
ax14.tick_params(labelsize=7)
cbar14 = plt.colorbar(im14, ax=ax14, fraction=0.046, pad=0.04)
cbar14.ax.tick_params(labelsize=7)

ax15 = fig.add_subplot(gs[3, 2])
im15 = ax15.imshow(R_thresh, cmap='gray', interpolation='nearest')
ax15.set_title(f'R (thresholded)\nT={T:.2f}', fontsize=12, fontweight='bold', pad=10)
ax15.set_xlabel('Column', fontsize=9)
ax15.set_ylabel('Row', fontsize=9)
ax15.tick_params(labelsize=7)
cbar15 = plt.colorbar(im15, ax=ax15, fraction=0.046, pad=0.04)
cbar15.ax.tick_params(labelsize=7)

ax16 = fig.add_subplot(gs[3, 3])
im16 = ax16.imshow(R_nms, cmap='gray', interpolation='nearest')
ax16.set_title(f'R (after NMS)\n{int(np.count_nonzero(R_nms))} corners', fontsize=12, fontweight='bold', pad=10)
ax16.set_xlabel('Column', fontsize=9)
ax16.set_ylabel('Row', fontsize=9)
ax16.tick_params(labelsize=7)
cbar16 = plt.colorbar(im16, ax=ax16, fraction=0.046, pad=0.04)
cbar16.ax.tick_params(labelsize=7)

fig.suptitle('Harris-Stephens Corner Detection', 
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('harris_pipeline.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: harris_pipeline.png")

# Create a second figure showing final result with detected corners overlaid
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))
fig2.patch.set_facecolor('white')

# Original image with corners marked
axes2[0].imshow(img, cmap='gray', interpolation='nearest')
if corners:
    corner_rows = [c[0] for c in corners]
    corner_cols = [c[1] for c in corners]
    axes2[0].plot(corner_cols, corner_rows, 'r+', markersize=20, markeredgewidth=3, label='Detected Corners')
    axes2[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes2[0].set_title(f'Original Image with {len(corners)} Detected Corners', fontsize=13, fontweight='bold', pad=12)
axes2[0].set_xlabel('Column', fontsize=10)
axes2[0].set_ylabel('Row', fontsize=10)
axes2[0].tick_params(labelsize=9)
axes2[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Response map (scaled) with corners overlaid
axes2[1].imshow(R_scaled, cmap='gray', interpolation='nearest')
if corners:
    # Add white circles around corners for better visibility
    for row, col, _ in corners:
        circle = plt.Circle((col, row), 0.6, color='red', fill=False, linewidth=3)
        axes2[1].add_patch(circle)
    axes2[1].plot(corner_cols, corner_rows, 'r+', markersize=20, markeredgewidth=3, label='Corners')
    axes2[1].legend(loc='upper right', fontsize=11, framealpha=0.9)
axes2[1].set_title('Response Map (Scaled)', fontsize=13, fontweight='bold', pad=12)
axes2[1].set_xlabel('Column', fontsize=10)
axes2[1].set_ylabel('Row', fontsize=10)
axes2[1].tick_params(labelsize=9)
axes2[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Final NMS result
im_nms = axes2[2].imshow(R_nms, cmap='gray', interpolation='nearest')
axes2[2].set_title('After Non-Maximum Suppression', fontsize=13, fontweight='bold', pad=12)
axes2[2].set_xlabel('Column', fontsize=10)
axes2[2].set_ylabel('Row', fontsize=10)
axes2[2].tick_params(labelsize=9)
axes2[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add colorbar to last subplot
cbar = plt.colorbar(im_nms, ax=axes2[2], fraction=0.046, pad=0.04)
cbar.set_label('Corner Response', rotation=270, labelpad=20, fontsize=10)
cbar.ax.tick_params(labelsize=9)

fig2.suptitle('Harris Corner Detection: Final Results', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('harris_final_results.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: harris_final_results.png")

# Create a third figure showing corner details
if corners:
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    fig3.patch.set_facecolor('white')
    
    # Create a copy of the image for annotation
    img_annotated = np.copy(img)
    
    # Plot image
    ax3.imshow(img_annotated, cmap='gray', interpolation='nearest')
    
    # Mark corners with circles and labels
    for idx, (row, col, val) in enumerate(corners):
        # Draw circle around corner
        circle = plt.Circle((col, row), 0.9, color='black', fill=False, linewidth=3)
        ax3.add_patch(circle)
        # Add corner marker
        ax3.plot(col, row, 'k+', markersize=18, markeredgewidth=3)
        # Add label
        ax3.text(col, row-1.5, f'{idx+1}', color='white', fontsize=12, 
                fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='white', linewidth=2))
    
    ax3.set_title(f'Detected Corners with Labels (Total: {len(corners)})', 
                 fontsize=15, fontweight='bold', pad=15)
    ax3.set_xlabel('Column', fontsize=11)
    ax3.set_ylabel('Row', fontsize=11)
    ax3.tick_params(labelsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add text box with corner information
    corner_text = "Corner Details:\n" + "="*42 + "\n"
    for idx, (row, col, val) in enumerate(corners):
        corner_text += f"#{idx+1}: (row={row:2d}, col={col:2d}, R={val:7d})\n"
    corner_text += "="*42
    
    ax3.text(0.02, 0.98, corner_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, 
                     edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('harris_corners_labeled.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("✓ Saved: harris_corners_labeled.png")
    
    plt.close(fig3)

# Create a fourth figure: 3D visualization of response map
fig4 = plt.figure(figsize=(18, 8))
fig4.patch.set_facecolor('white')

# 3D surface plot of raw response
ax_3d1 = fig4.add_subplot(121, projection='3d')
ax_3d1.set_facecolor('white')
x_coords = np.arange(R_raw.shape[1])
y_coords = np.arange(R_raw.shape[0])
X, Y = np.meshgrid(x_coords, y_coords)
surf1 = ax_3d1.plot_surface(X, Y, R_raw, cmap='gray', alpha=0.9, 
                            linewidth=0, antialiased=True, edgecolor='none')
ax_3d1.set_xlabel('Column (x)', fontsize=11, labelpad=10)
ax_3d1.set_ylabel('Row (y)', fontsize=11, labelpad=10)
ax_3d1.set_zlabel('Response Value', fontsize=11, labelpad=10)
ax_3d1.set_title('3D Response Map (Raw)', fontsize=13, fontweight='bold', pad=15)
ax_3d1.tick_params(labelsize=9)
ax_3d1.view_init(elev=25, azim=45)
cbar1 = fig4.colorbar(surf1, ax=ax_3d1, shrink=0.5, aspect=10, pad=0.1)
cbar1.ax.tick_params(labelsize=9)

# 3D scatter plot showing detected corners
ax_3d2 = fig4.add_subplot(122, projection='3d')
ax_3d2.set_facecolor('white')
surf2 = ax_3d2.plot_surface(X, Y, R_raw, cmap='gray', alpha=0.4,
                            linewidth=0, antialiased=True, edgecolor='none')
if corners:
    corner_rows_3d = [c[0] for c in corners]
    corner_cols_3d = [c[1] for c in corners]
    corner_vals_3d = [c[2] for c in corners]
    ax_3d2.scatter(corner_cols_3d, corner_rows_3d, corner_vals_3d, 
                   c='black', marker='o', s=150, edgecolors='red', linewidth=3,
                   label='Detected Corners', depthshade=True)
    # Add vertical lines to show corner positions
    for col, row, val in zip(corner_cols_3d, corner_rows_3d, corner_vals_3d):
        ax_3d2.plot([col, col], [row, row], [0, val], 'r--', linewidth=1.5, alpha=0.6)
    ax_3d2.legend(fontsize=10, loc='upper left')
ax_3d2.set_xlabel('Column (x)', fontsize=11, labelpad=10)
ax_3d2.set_ylabel('Row (y)', fontsize=11, labelpad=10)
ax_3d2.set_zlabel('Response Value', fontsize=11, labelpad=10)
ax_3d2.set_title('3D View with Detected Corners', fontsize=13, fontweight='bold', pad=15)
ax_3d2.tick_params(labelsize=9)
ax_3d2.view_init(elev=25, azim=45)

fig4.suptitle('Harris Response Map: 3D Visualization', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('harris_3d_visualization.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: harris_3d_visualization.png")

plt.close('all')

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print(f"  📄 {OUT_TXT} - Complete text output with all calculations")
print("  📊 harris_pipeline.png - Complete pipeline visualization (16 subplots)")
print("  📊 harris_final_results.png - Final results with corners marked")
if corners:
    print("  📊 harris_corners_labeled.png - Labeled corner locations (grayscale)")
print("  📊 harris_3d_visualization.png - 3D response map (grayscale)")
print("\n" + "="*80)
print("Format: All images are in black and white (grayscale) with improved layout")
print("Resolution: 200 DPI for high-quality output")
print("="*80)
