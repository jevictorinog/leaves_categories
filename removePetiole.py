import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import label
from pylab import *
from imageUtils import *


def smooth_array(A, f):
    for i in range(A.size-1):
        A[i+1] = f*A[i+1] + (1-f)*A[i]

def last_local_min(perf):
    discr = np.rint(perf*100)/100
    ndx = []
    for i in range(discr.size-1):
        if discr[i] != discr[i+1]:
            ndx.append(i)
    if len(ndx) == 0:
        print('len perf3 petiole:', len(perf), 'len ndx:', len(ndx))
        return len(discr)-1
    pos = ndx[-1]
    for i in range(1, len(ndx)-1):
        if discr[ndx[i-1]] > discr[ndx[i]] < discr[ndx[i+1]]:
            pos = ndx[i]
    high = 0
    for i in range(perf.size-8):
        if perf[i] < 0.1 and (perf[i+8]-perf[i]) > 0.5:
            high = i
    if high > 0: pos = high
    return pos

def get_larger_region(binImg):
    imlb, num = label(binImg)
    mmax = 0
    for i in range(1,num+1):
        mm = np.count_nonzero(imlb==i)
        if mm>mmax:
            mmax = mm
            ndx = i
    return uint8((imlb==ndx)*255)


def removePetiole(imb):
    alto = imb.shape[0]
    perf1 = imb.sum(axis=1)
    perf1 = perf1[::-1]/255
    perf2 = perf1.copy()
    smooth_array(perf2, 1/16)
    bajo = np.max(perf2)*0.25
    perf3 = np.diff(perf2)
    smooth_array(perf3, 1/16)

    mk = np.array([0]*5)
    i = 0
    while perf1[i] == 0: i+=1
    mk[0] = i

    mk2 = mk[0]
    while perf1[mk2] < bajo: mk2+=1

    mk[1] = last_local_min(perf3[mk[0]:mk2+1]) + mk[0]

    i = alto-1
    while perf1[i] == 0: i-=1
    mk[2] = i
    mk[3] = (mk[2]-mk[1])//2 + mk[1]
    mk[4] = mk[2] - (mk[2]-mk[1])//2

    mk = alto - mk
    if mk[1]+2 >= alto or mk[0]+3 >= alto: mk[1]=alto-3; mk[0]=alto-1

    return mk


def bodyAnalysis(contour):
    sz_contour = 1025
    j = (-1)**0.5
    f1 = 6
    x = contour[:,0] + contour[:,1]*j;
    ndk1 = np.fft.fft(x.T)
    ndk2 = ndk1.copy()
    ndk3 = ndk1.copy()
    ndk1[f1:sz_contour-f1]  = 0
    ndk2[16:sz_contour-16]  = 0
    ndk3[50:sz_contour-50]  = 0
    y1 = np.fft.ifft(ndk1)
    y2 = np.fft.ifft(ndk2)
    y3 = np.fft.ifft(ndk3)
    r1 = (y1-y2)
    r2 = (y3-y2)

    return y1, r1, r2


def verticalPca(feat):
    pca = PCA(n_components=2)
    pca.fit(feat)
    pc1 = pca.components_[0,:]
    ang = np.arctan(pc1[1]/pc1[0])*180/np.pi
    if pc1[0] > 0 :
        if pc1[1] > 0 : qua = 1
        else: qua = 4; ang = 180+ang
    else:
        if pc1[1] > 0: qua = 2; ang = 180+ang
        else: qua = 3; ang = ang

    return ang

def preprocess(imFilename):
    imb = binImage(imFilename)
    outline = getContour(imb)
    ang = verticalPca(outline)
    rot = ang-90 if 65<ang<115 else 0
    im1 = rotateBinImage(imb, -rot)
    marks = removePetiole(im1)
    imbody = im1[marks[2]-2:marks[1]+2, :]
    outline = getContour(imbody)
    contour = interparc(1025, outline)
    return bodyAnalysis(contour)


#remove_petiole('/home/jorge/data/leaf_image/ImageCLEF2012/data/bin_species/Buxus_sempervirens/6743.png',1)
