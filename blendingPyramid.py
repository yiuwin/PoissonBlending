# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 14:59:50 2016

@author: zhouc
"""
import numpy as np
import cv2
import time
import math 

def createPyramid(img, pyramidN):

    imagePyramid = list()
    gaussianPyramid = list()
    laplacePyramid = list()

    h,w = img.shape[:2]
    imagePyramid.append(img)

    for i in xrange(int(pyramidN)):
        down = cv2.resize(img, (w/2,h/2))
        imagePyramid.append(down)
        up = cv2.resize(down,(w,h))
        gaussianPyramid.append(up)
        lp = imagePyramid[i]-gaussianPyramid[i]

        laplacePyramid.append(lp)
        img = down
        h,w = img.shape[:2] # size down h,w for next round

    return imagePyramid, gaussianPyramid, laplacePyramid


# config & input
start = time.time()

Topic = 'apple'

backImageName = Topic + '.png'
foreImageName = Topic + '2.png'
maskName = Topic + '_mask.png'
outputName = Topic + '_out_pyramid.png'


backImg = cv2.imread(backImageName) / 255.0
foreImg = cv2.imread(foreImageName) / 255.0
mask = cv2.imread(maskName) / 255.0

rows = backImg.shape[0]
cols = backImg.shape[1]
channels = backImg.shape[2]

if mask.ndim == 2:
    mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])

if mask.shape[2] == 1:
    mask = np.tile(mask, [1, 1, 3])

pyramidN = math.ceil(math.log(min(rows, cols) / 16, 2))

# build pyramid
[imageFore, gaussianFore, laplaceFore] = createPyramid(foreImg, pyramidN)
[imageBack, gaussianBack, laplaceBack] = createPyramid(backImg, pyramidN)
[imageMask, gaussianMask, laplaceMask] = createPyramid(mask, pyramidN)


"""
TODO 2
Combine the laplacian pyramids of background and foreground

add your code here
"""

laplaceMerge = list()

for mergeLevel in range(len(gaussianMask)):
    a = gaussianMask[mergeLevel]*laplaceFore[mergeLevel]
    b = (1-gaussianMask[mergeLevel])*laplaceBack[mergeLevel]
    laplaceMerge.append(a + b)

# Combine the smallest scale image

"""
TODO 3
Combine the smallest scale images of background and foreground

add your code here
"""

smallestScale = ( imageFore[-2]*gaussianMask[-1] ) + ( imageBack[-2]*(1-gaussianMask[-1]) )

# reconstruct & output

"""
TODO 4
reconstruct the blending image by adding the gradient (in different scale) back to
the smallest scale image while upsampling

add your code here
"""

for reconstruct in xrange(len(laplaceMerge)-2,-1,-1):
    smallestScale = cv2.resize(smallestScale, (laplaceMerge[reconstruct].shape[0],laplaceMerge[reconstruct].shape[1]))
    img = smallestScale + laplaceMerge[reconstruct]
    smallestScale = img

#cv2.imshow("back", backImg)
#cv2.imshow("fore",foreImg)
cv2.imshow('output', img);
#print (time.time() - start)
cv2.waitKey(0)
cv2.imwrite(outputName, img * 255);
