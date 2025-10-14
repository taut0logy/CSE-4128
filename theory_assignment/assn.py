import os
import cv2
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------
# ---- starter image  ----
arr = [[58, 59, 59, 64, 58, 61, 58, 51, 50, 51, 51, 54, 49],
    [59, 60, 57, 58, 58, 58, 58, 53, 50, 51, 51, 52, 54],
    [58, 61, 61, 60, 61, 57, 61, 49, 51, 53, 53, 51, 52],
    [57, 60, 62, 63, 58, 57, 199, 54, 50, 51, 53, 50, 52],
    [58, 57, 57, 59, 58, 196, 193, 194, 54, 50, 51, 51, 53],
    [57, 57, 60, 62, 194, 194, 195, 198, 195, 198, 50, 54, 50],
    [17, 19, 17, 23, 194, 196, 193, 194, 194, 197, 51, 51, 54],
    [16, 19, 17, 21, 194, 194, 194, 195, 194, 197, 53, 52, 53],
    [17, 18, 19, 21, 195, 193, 199, 198, 197, 195, 52, 52, 49],
    [16, 20, 18, 20, 194, 193, 194, 197, 196, 194, 50, 52, 53],
    [17, 18, 21, 21, 194, 195, 197, 193, 195, 194, 51, 52, 52],
    [16, 19, 20, 17, 84, 85, 87, 85, 86, 87, 51, 52, 51],
    [17, 16, 21, 88, 85, 84, 85, 86, 86, 87, 50, 50, 51]]

img = np.array(arr, dtype=np.float32)
# ---------------------------

RES_DIR = "output/"
os.makedirs(RES_DIR, exist_ok=True)

# Output filename
OUT_TXT = f"{RES_DIR}/result.txt"

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

# Create a comprehensive figure with all intermediate results
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

# Row 1: Input and Derivative Kernels
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(img, cmap='gray', interpolation='nearest')
ax1.set_title('Input Image', fontsize=11, fontweight='bold')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(kernel_dx, cmap='seismic', interpolation='nearest', vmin=-0.5, vmax=0.5)
ax2.set_title('Derivative Kernel Kx\n(3x3, σ=0.5)', fontsize=10, fontweight='bold')
for i in range(deriv_m):
    for j in range(deriv_m):
        ax2.text(j, i, f'{kernel_dx[i,j]:.3f}', ha='center', va='center', fontsize=7, color='black')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046)

ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(kernel_dy, cmap='seismic', interpolation='nearest', vmin=-0.5, vmax=0.5)
ax3.set_title('Derivative Kernel Ky\n(3x3, σ=0.5)', fontsize=10, fontweight='bold')
for i in range(deriv_m):
    for j in range(deriv_m):
        ax3.text(j, i, f'{kernel_dy[i,j]:.3f}', ha='center', va='center', fontsize=7, color='black')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, fraction=0.046)

ax4 = fig.add_subplot(gs[0, 3])
im4 = ax4.imshow(gaussian_window, cmap='viridis', interpolation='nearest')
ax4.set_title('Gaussian Window\n(3x3, σ=0.6)', fontsize=10, fontweight='bold')
for i in range(win_m):
    for j in range(win_m):
        ax4.text(j, i, f'{gaussian_window[i,j]:.3f}', ha='center', va='center', fontsize=7, color='white')
ax4.axis('off')
plt.colorbar(im4, ax=ax4, fraction=0.046)

# Row 2: Gradients and Their Squares
ax5 = fig.add_subplot(gs[1, 0])
im5 = ax5.imshow(Ix, cmap='seismic', interpolation='nearest')
ax5.set_title('Ix (floored)', fontsize=11, fontweight='bold')
ax5.axis('off')
plt.colorbar(im5, ax=ax5, fraction=0.046)

ax6 = fig.add_subplot(gs[1, 1])
im6 = ax6.imshow(Iy, cmap='seismic', interpolation='nearest')
ax6.set_title('Iy (floored)', fontsize=11, fontweight='bold')
ax6.axis('off')
plt.colorbar(im6, ax=ax6, fraction=0.046)

ax7 = fig.add_subplot(gs[1, 2])
im7 = ax7.imshow(Ix2, cmap='hot', interpolation='nearest')
ax7.set_title('Ix² (before smoothing)', fontsize=11, fontweight='bold')
ax7.axis('off')
plt.colorbar(im7, ax=ax7, fraction=0.046)

ax8 = fig.add_subplot(gs[1, 3])
im8 = ax8.imshow(Iy2, cmap='hot', interpolation='nearest')
ax8.set_title('Iy² (before smoothing)', fontsize=11, fontweight='bold')
ax8.axis('off')
plt.colorbar(im8, ax=ax8, fraction=0.046)

# Row 3: Smoothed Components and Structure Tensor
ax9 = fig.add_subplot(gs[2, 0])
im9 = ax9.imshow(Ix2_smooth, cmap='hot', interpolation='nearest')
ax9.set_title('Ix² (smoothed)', fontsize=11, fontweight='bold')
ax9.axis('off')
plt.colorbar(im9, ax=ax9, fraction=0.046)

ax10 = fig.add_subplot(gs[2, 1])
im10 = ax10.imshow(Iy2_smooth, cmap='hot', interpolation='nearest')
ax10.set_title('Iy² (smoothed)', fontsize=11, fontweight='bold')
ax10.axis('off')
plt.colorbar(im10, ax=ax10, fraction=0.046)

ax11 = fig.add_subplot(gs[2, 2])
im11 = ax11.imshow(Ixy_smooth, cmap='seismic', interpolation='nearest')
ax11.set_title('IxIy (smoothed)', fontsize=11, fontweight='bold')
ax11.axis('off')
plt.colorbar(im11, ax=ax11, fraction=0.046)

ax12 = fig.add_subplot(gs[2, 3])
im12 = ax12.imshow(detM, cmap='viridis', interpolation='nearest')
ax12.set_title('det(M)', fontsize=11, fontweight='bold')
ax12.axis('off')
plt.colorbar(im12, ax=ax12, fraction=0.046)

# Row 4: Response Map and Final Results
ax13 = fig.add_subplot(gs[3, 0])
im13 = ax13.imshow(R_raw, cmap='jet', interpolation='nearest')
ax13.set_title(f'R (raw)\nκ={kappa}', fontsize=11, fontweight='bold')
ax13.axis('off')
plt.colorbar(im13, ax=ax13, fraction=0.046)

ax14 = fig.add_subplot(gs[3, 1])
im14 = ax14.imshow(R_scaled, cmap='jet', interpolation='nearest')
ax14.set_title('R (scaled 0-255)', fontsize=11, fontweight='bold')
ax14.axis('off')
plt.colorbar(im14, ax=ax14, fraction=0.046)

ax15 = fig.add_subplot(gs[3, 2])
im15 = ax15.imshow(R_thresh, cmap='jet', interpolation='nearest')
ax15.set_title(f'R (thresholded)\nT={T:.2f}', fontsize=11, fontweight='bold')
ax15.axis('off')
plt.colorbar(im15, ax=ax15, fraction=0.046)

ax16 = fig.add_subplot(gs[3, 3])
im16 = ax16.imshow(R_nms, cmap='jet', interpolation='nearest')
ax16.set_title(f'R (after NMS)\n{int(np.count_nonzero(R_nms))} corners', fontsize=11, fontweight='bold')
ax16.axis('off')
plt.colorbar(im16, ax=ax16, fraction=0.046)

fig.suptitle('Harris-Stephens Corner Detection: Complete Pipeline', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f'{RES_DIR}/harris_pipeline.png', dpi=150, bbox_inches='tight')
print("✓ Saved: harris_pipeline.png")

# Create a second figure showing final result with detected corners overlaid
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# Original image with corners marked
axes2[0].imshow(img, cmap='gray', interpolation='nearest')
if corners:
    corner_rows = [c[0] for c in corners]
    corner_cols = [c[1] for c in corners]
    axes2[0].plot(corner_cols, corner_rows, 'r+', markersize=15, markeredgewidth=2, label='Detected Corners')
    axes2[0].legend(loc='upper right', fontsize=10)
axes2[0].set_title(f'Original Image with {len(corners)} Detected Corners', fontsize=12, fontweight='bold')
axes2[0].axis('off')

# Response map (scaled) with corners overlaid
axes2[1].imshow(R_scaled, cmap='jet', interpolation='nearest')
if corners:
    axes2[1].plot(corner_cols, corner_rows, 'w+', markersize=15, markeredgewidth=3, label='Corners')
    axes2[1].legend(loc='upper right', fontsize=10)
axes2[1].set_title('Response Map (Scaled)', fontsize=12, fontweight='bold')
axes2[1].axis('off')

# Final NMS result
axes2[2].imshow(R_nms, cmap='jet', interpolation='nearest')
axes2[2].set_title('After Non-Maximum Suppression', fontsize=12, fontweight='bold')
axes2[2].axis('off')

# Add colorbar to last subplot
cbar = plt.colorbar(axes2[2].images[0], ax=axes2[2], fraction=0.046)
cbar.set_label('Corner Response', rotation=270, labelpad=15)

fig2.suptitle('Harris Corner Detection: Final Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RES_DIR}/harris_final_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: harris_final_results.png")

# Create a third figure showing corner details
if corners:
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    # Create a copy of the image for annotation
    img_annotated = np.copy(img)
    
    # Plot image
    ax3.imshow(img_annotated, cmap='gray', interpolation='nearest')
    
    # Mark corners with circles and labels
    for idx, (row, col, val) in enumerate(corners):
        circle = plt.Circle((col, row), 0.8, color='red', fill=False, linewidth=2)
        ax3.add_patch(circle)
        ax3.text(col, row-1.2, f'{idx+1}', color='yellow', fontsize=10, 
                fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    ax3.set_title(f'Detected Corners with Labels (Total: {len(corners)})', 
                 fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add text box with corner information
    corner_text = "Corner Details:\n" + "-"*40 + "\n"
    for idx, (row, col, val) in enumerate(corners):
        corner_text += f"#{idx+1}: (row={row}, col={col}, R={val})\n"
    
    ax3.text(0.02, 0.98, corner_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{RES_DIR}/harris_corners_labeled.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: harris_corners_labeled.png")
    
    plt.close(fig3)

# Create a fourth figure: 3D visualization of response map
fig4 = plt.figure(figsize=(14, 6))

# 3D surface plot of raw response
ax_3d1 = fig4.add_subplot(121, projection='3d')
x_coords = np.arange(R_raw.shape[1])
y_coords = np.arange(R_raw.shape[0])
X, Y = np.meshgrid(x_coords, y_coords)
surf1 = ax_3d1.plot_surface(X, Y, R_raw, cmap='viridis', alpha=0.8)
ax_3d1.set_xlabel('Column (x)')
ax_3d1.set_ylabel('Row (y)')
ax_3d1.set_zlabel('Response Value')
ax_3d1.set_title('3D Response Map (Raw)', fontsize=12, fontweight='bold')
fig4.colorbar(surf1, ax=ax_3d1, shrink=0.5)

# 3D scatter plot showing detected corners
ax_3d2 = fig4.add_subplot(122, projection='3d')
ax_3d2.plot_surface(X, Y, R_raw, cmap='viridis', alpha=0.3)
if corners:
    corner_rows_3d = [c[0] for c in corners]
    corner_cols_3d = [c[1] for c in corners]
    corner_vals_3d = [c[2] for c in corners]
    ax_3d2.scatter(corner_cols_3d, corner_rows_3d, corner_vals_3d, 
                   c='red', marker='o', s=100, edgecolors='darkred', linewidth=2,
                   label='Detected Corners')
    ax_3d2.legend()
ax_3d2.set_xlabel('Column (x)')
ax_3d2.set_ylabel('Row (y)')
ax_3d2.set_zlabel('Response Value')
ax_3d2.set_title('3D View with Detected Corners', fontsize=12, fontweight='bold')

fig4.suptitle('Harris Response Map: 3D Visualization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RES_DIR}/harris_3d_visualization.png', dpi=150, bbox_inches='tight')
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
    print("  📊 harris_corners_labeled.png - Labeled corner locations")
print("  📊 harris_3d_visualization.png - 3D response map visualization")
print("="*80)
