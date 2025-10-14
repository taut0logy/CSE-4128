import numpy as np
import cv2
from tabulate import tabulate

def area(img):
    return np.count_nonzero(img)

def perimeter(img):
    se=np.zeros((3,3), dtype=np.uint8)
    eroded=cv2.erode(img,se,iterations=1)
    border=img-eroded
    return np.count_nonzero(border)

def axes(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse
        a = max(MA, ma)
        b = min(MA, ma)
        return a, b
    else:
        return 0, 0

def features(img):
    A=area(img)
    P=perimeter(img)
    a, b=axes(img)
    
    compactness=P**2/A
    form_factor=4*np.pi*A/P**2
    eccentricity=np.sqrt(1-(b**2/a**2)) if a!=0 else 0
    
    return compactness,form_factor,eccentricity

def kullback_leibler(img1, img2):
    c1, f1, r1 = features(img1)
    c2, f2, r2 = features(img2)

    p = np.array([c1, f1, r1])
    q = np.array([c2, f2, r2])
    
    ph = p / np.sum(p)
    qh = q / np.sum(q)
    
    d = 0
    for i in range(len(ph)):
        if ph[i] != 0 and qh[i] != 0:
            d += ph[i] * np.log(ph[i] / qh[i])
    
    return d

gt1=cv2.imread('../../assets/train1.jpg', cv2.IMREAD_GRAYSCALE)
gt2=cv2.imread('../../assets/train2.jpg', cv2.IMREAD_GRAYSCALE)
gt3=cv2.imread('../../assets/train3.png', cv2.IMREAD_GRAYSCALE)
t1=cv2.imread('../../assets/train1.jpg', cv2.IMREAD_GRAYSCALE)
t2=cv2.imread('../../assets/train2.jpg', cv2.IMREAD_GRAYSCALE)
t3=cv2.imread('../../assets/train3.png', cv2.IMREAD_GRAYSCALE)
t4=cv2.imread('../../assets/t1.jpg', cv2.IMREAD_GRAYSCALE)

train=[gt1,gt2,gt3]
test=[t1,t2,t3,t4]

mat=np.zeros((4,3), dtype=np.float32)

for i,x in enumerate(test):
    for j,y in enumerate(train):
        mat[i][j] = kullback_leibler(x, y)

mat=np.array(mat)

row_headers = [f'Test {i + 1}' for i in range(4)]
col_headers = [f'GT {i + 1}' for i in range(3)]

print(tabulate(mat[0:4,0:3], headers=col_headers, showindex=row_headers, tablefmt='grid'))