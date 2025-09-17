import cv2
import numpy as np

def region_descriptors(binary_mask):
    # Find contours of the object
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]  # take largest contour if multiple
    
    # Basic properties
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Compactness (P^2 / A)
    compactness = (perimeter**2) / area if area > 0 else 0
    
    # Form Factor (4πA / P^2)
    form_factor = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    
    # Bounding box for elongation
    x, y, w, h = cv2.boundingRect(cnt)
    elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    
    # Fit ellipse (for roundness & eccentricity)
    if len(cnt) >= 5:  # need at least 5 points
        ellipse = cv2.fitEllipse(cnt)
        (xc, yc), (MA, ma), angle = ellipse  # MA = minor axis, ma = major axis
        roundness = (4 * area) / (np.pi * (ma**2)) if ma > 0 else 0
        eccentricity = np.sqrt(1 - (MA/ma)**2) if ma > 0 else 0
    else:
        roundness, eccentricity = 0, 0

    return {
        "Area": area,
        "Perimeter": perimeter,
        "Compactness": compactness,
        "Form Factor": form_factor,
        "Elongation": elongation,
        "Roundness": roundness,
        "Eccentricity": eccentricity
    }

# ------------------------------
# Test with simple shapes
# ------------------------------
shapes = {
    "Circle": cv2.circle(np.zeros((200,200), dtype=np.uint8), (100,100), 50, 255, -1),
    "Rectangle": cv2.rectangle(np.zeros((200,200), dtype=np.uint8), (50,50), (150,120), 255, -1),
    "Ellipse": cv2.ellipse(np.zeros((200,200), dtype=np.uint8), (100,100), (60,30), 0, 0, 360, 255, -1)
}

for name, mask in shapes.items():
    print(f"\n{name} descriptors:")
    desc = region_descriptors(mask)
    for k, v in desc.items():
        print(f"  {k}: {v:.4f}")
