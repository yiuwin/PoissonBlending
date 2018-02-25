from __future__ import print_function

import numpy as np
import scipy
import math
from scipy.sparse import linalg
from scipy.misc import toimage
from scipy.misc import imshow
import cv2
from PIL import Image
import time

# config & input
start = time.time()

def generateData(back,fore,mask):
    I = []
    J = []
    S = []
    B = []
    count = 0

    for i in xrange(int(back.shape[0])):
        for j in xrange(int(back.shape[1])):
            if mask.item(i,j) < 0.5: #black pixel, then insert 1 at that index
                I.extend([count])
                J.extend([count])
                S.extend([1])
                B.extend([back[i,j]]) #set b = background pixel value
            else: #white pixel, insert gradient of i,j
                J.extend([count-1, count+1, count-fore.shape[1], count+fore.shape[1], count])
                I.extend([count,   count,   count,               count,               count])
                S.extend([1,       1,       1,                   1,                   -4])

                B.extend( [fore[i-1,j] + fore[i+1,j] + fore[i,j-1] + fore[i,j+1] - 4.0*fore[i,j]] )
                
                ''' 
                With Gradient Mixing 
                tmpforeB = fore[i-1,j] + fore[i+1,j] + fore[i,j-1] + fore[i,j+1] - 4.0*fore[i,j]
                tmpbackB = back[i-1,j] + back[i+1,j] + back[i,j-1] + back[i,j+1] - 4.0*back[i,j]
                B.extend( [0.5*tmpforeB + 0.5*tmpbackB] )
                '''

            count+=1

    I = np.asarray(I) #column for SPARSE MATRIX A
    J = np.asarray(J) #row for SPARSE MATRIX A
    S = np.asarray(S) #data for SPARSE MATRIX A
    B = np.asarray(B) #B for Ax=b

    return I,J,S,B

''' genereateAB using zero ndarray : TOO MUCH MEMORY
def generateAB(back,fore,mask,alls):
    A = np.zeros(shape =(alls,alls))
    B = []
    for i in xrange(int(back.shape[0])):
        for j in xrange(int(back.shape[1])):
            if mask.item(i,j) < 0.5:
                A.itemset((i,j),1)
                B.append(back[i,j])
            else:
                A.itemset((i,j),-4)
                A.itemset((i,j-4),1)
                A.itemset((i,j-1),1)
                A.itemset((i,j+1),1)
                A.itemset((i,j+4),1)
                B.append( [fore[i-1,j] + fore[i,j-1] + fore[i,j+1] + fore[i+1,j] - 4*(fore[i,j])] )
    B = np.asarray(B)
    return A, B
'''

# Topic = 'snow'
Topic = 'notebook'


backImageName = Topic + '.png'
foreImageName = Topic + '2.png'
maskName = Topic + '_mask.png'
outputName = Topic + '_out_poisson.png'


backImg = cv2.imread(backImageName) / 255.0
foreImg = cv2.imread(foreImageName) / 255.0
mask = cv2.imread(maskName,0) / 255.0

rows = backImg.shape[0] #545p
cols = backImg.shape[1] #429p
channels = backImg.shape[2] #3 for BGR

#alls = rows * cols * channels
alls = rows * cols #total number of pixels in image

#split BGR
backB, backG, backR = cv2.split(backImg)
foreB, foreG, foreR = cv2.split(foreImg)

#print (backImg)


''' TEST NDARRAY ITEMSET
B = np.zeros(4)
B.itemset(1,2)
print B
'''

''' TESTING R WITH SMALL NDARRAY 
print ("***** Testing R with Small NDARRAY *****")
TestA = np.array([[0,0,1,1],[1,0,2,0],[1,1,0,1],[1,1,0,0]])
TestB = np.array([15,12,22,16])

print ("matrix A at [2,1]",end ="")
print (TestA[2,1])

TestA = scipy.sparse.coo_matrix(TestA) #convert np.array A to coo_matrix
TestA = TestA.tocsc()

print ("matrix A shape: ",end="")
print (TestA.shape)
print (TestA)
print ("matrix B shape: %s" % TestB.shape)
print (TestB)
R = scipy.sparse.linalg.spsolve(TestA,TestB)
print ("solution X type: %s" % type(R))
print (R)
print ("solution X shape: ",end="")
print (R.shape)
print ("\n\n")
'''

"""
Construct matrix A & B
"""
print ("***** Generating Matrices Ab, Ag, Ar *****")

numRowsInA = alls # pixels(row) * pixels(col)

Ib,Jb,Sb,Bb = generateData(backB,foreB,mask)
Ig, Jg, Sg, Bg = generateData(backG,foreG,mask)
Ir, Jr, Sr, Br = generateData(backR,foreR,mask)

Ab = scipy.sparse.coo_matrix((Sb, (Ib, Jb)), shape=(numRowsInA, alls))
Ag = scipy.sparse.coo_matrix((Sg, (Ig, Jg)), shape=(numRowsInA, alls))
Ar = scipy.sparse.coo_matrix((Sr, (Ir, Jr)), shape=(numRowsInA, alls))
Ab = Ab.tocsc() # Convert A matrix to Compressed Sparse Row format
Ag = Ag.tocsc()
Ar = Ar.tocsc()

"""
extract final result from R
Solve Ax = b for each of B,G,R
"""

print ("***** Solving X for AX = B *****")
#R = scipy.sparse.linalg.cg(Ab, Bb)
Rb = scipy.sparse.linalg.spsolve(Ab,Bb)
Rb = np.reshape(Rb, (rows,cols))
Rg = scipy.sparse.linalg.spsolve(Ag,Bg)
Rg = np.reshape(Rg, (rows,cols))
Rr = scipy.sparse.linalg.spsolve(Ar,Br)
Rr = np.reshape(Rr, (rows,cols))

merged = cv2.merge((Rb,Rg,Rr))
cv2.imshow("merged",merged)
#print (time.time() - start)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
uncomment these lines after you generate the final result in matrix 'img'
cv2.imshow('output', R);
cv2.waitKey(0)
cv2.imwrite(outputName, R * 255);
"""
