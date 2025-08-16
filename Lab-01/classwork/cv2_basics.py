# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2

path='raufun.jpg'

# Read image in color
img_color = cv2.imread(path, cv2.IMREAD_COLOR) # 1, only color without alpha



# Read image in grayscale
img_grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 0 

img_orig = cv2.imshow(path, cv2.IMREAD_UNCHANGED) # -1, umchanged. Includes alpha

cv2.imshow("Color image", img_color)
cv2.imshow("Grey image", img_grey)


cv2.waitKey(0)
cv2.destroyAllWindows()
