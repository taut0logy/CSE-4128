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

def dmax(img):
    xmin=img.shape[0]
    xmax=0
    ymin=img.shape[1]
    ymax=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]>=0:
                xmin=min(xmin,i)
                xmax=max(xmax,i)
                ymin=min(ymin,j)
                ymax=max(ymax,j)
    return max(xmax-xmin, ymax-ymin)

def features(img):
    A=area(img)
    P=perimeter(img)
    d=dmax(img)

    compactness=P**2/A
    form_factor=4*np.pi*A/P**2
    roundness=4*A/(np.pi*d**2)
    return compactness,form_factor,roundness

gt1=cv2.imread('../../assets/c1.jpg', cv2.IMREAD_GRAYSCALE)
gt2=cv2.imread('../../assets/t1.jpg', cv2.IMREAD_GRAYSCALE)
gt3=cv2.imread('../../assets/p1.png', cv2.IMREAD_GRAYSCALE)
t1=cv2.imread('../../assets/c2.jpg', cv2.IMREAD_GRAYSCALE)
t2=cv2.imread('../../assets/t2.jpg', cv2.IMREAD_GRAYSCALE)
t3=cv2.imread('../../assets/p2.png', cv2.IMREAD_GRAYSCALE)
t4=cv2.imread('../../assets/p3.jpg', cv2.IMREAD_GRAYSCALE)

train=[gt1,gt2,gt3]
test=[t1,t2,t3,t4]

mat=np.zeros((4,3), dtype=np.float32)

for i,x in enumerate(test):
    for j,y in enumerate(train):
        c1,f1,r1=features(x)
        c2,f2,r2=features(y)
        mat[i][j]=np.sqrt((c1-c2)**2+(f1-f2)**2+(r1-r2)**2)

mat=np.array(mat)

row_headers = [f'Test {i + 1}' for i in range(4)]
col_headers = [f'GT {i + 1}' for i in range(3)]

print(tabulate(mat[0:4,0:3], headers=col_headers, showindex=row_headers, tablefmt='grid'))