from imageUtils import *
from scipy.ndimage import label
import pandas as pd
import numpy as np
import pywt
import os

def load_data(path, lst_species):
    files        = []
    classLeaf   = []
    numbers     = []

    cl = 0
    for sp in lst_species:
        imgs = os.listdir(path + sp)
        for i in imgs:
            if i.endswith('.png'): 
                files.append(path + sp + '/' + i)
                numbers.append(i[:-4])
                classLeaf.append(cl)
        cl = cl+1

    df = pd.DataFrame(zip(files, numbers, classLeaf), columns=['file', 'num', 'cat'])
    df = df.sort_values('file')
    df.reset_index(inplace=True)
    return df

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

def rotateBinImage(imBin, ang):
    h = imBin.shape[0]
    w = imBin.shape[1]
    rot = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=ang, scale=1)
    return cv2.warpAffine(imBin, rot, dsize=(w, h))

def verticalStraighten(imb):
    outline = getContour(imb)
    ang = verticalPca(outline)
    rot = ang-90 if 65<ang<115 else 0
    if ang != 0:
        im1 = rotateBinImage(imb, -rot)
    else:
        im1 = imb
        print('No rotate', imFilename)
    return imb

def bigPatchImage(imb):
    imlb, num = label(imb)
    mmax = 0
    for i in range(1,num+1):
        mm = np.count_nonzero(imlb==i)
        if mm>mmax:
            mmax = mm
            ndx = i
    
    mask = imlb==ndx
    base = np.ones_like(imlb, np.uint8)

    imout = np.uint8((base & mask))*255
    return imout


def bodyAnalysis(imbody):
    f1, f2, f3 = 6, 16, 100
    sz_contour = 1025
    perf = imbody.sum(axis=1)
    outline = getContour(imbody)
    contour = interparc(1025, outline)
    x = contour[:,0] + contour[:,1]*1j;
    ndk1 = np.fft.fft(x.T)
    ndk2 = ndk1.copy()
    ndk3 = ndk1.copy()
    ndk1[f1:sz_contour-f1]  = 0
    ndk2[f2:sz_contour-f2]  = 0
    #ndk3[f3:sz_contour-f3]  = 0
    y1 = np.fft.ifft(ndk1)
    y2 = np.fft.ifft(ndk2)
    y3 = np.fft.ifft(ndk3)
    r1 = (y1-y2)
    r2 = (y3-y2)

    return y1, perf, r1, r2

def coef_ravel(coef):
    lst = []
    for c in coef:
        lst.extend(c)
    return np.array(lst)

def analysisBodyShapeFourier(contour, h=6):
    szc = contour.shape[0]-1
    ind_harm = list(range(h)) + list(range(szc-h, szc))
    x   = contour;
    dx  = np.diff(x, axis=0)
    ndx = np.divide(dx, np.absolute(dx))
    ndk = np.fft.fft(ndx.T)
    ndk = ndk.T
    return ndk[ind_harm]

def analysisBodyShapeWavelet(signal, h=256):
    szc = signal.shape[0]
    L = pywt.dwt_max_level(szc,'sym4')
    coef = pywt.wavedec(signal, 'sym4', level=L)
    rcoef = coef_ravel(coef)
    return rcoef[:h]

def analysisBodyProfileFourier(signal, num=256, h=6):
    ind_harm = list(range(h//2)) + list(range(num-h//2, num))
    ix = signal.mean()
    per2 = [signal[round(i*(len(signal)-1)/(num-1))]-ix for i in range(num)]
    ndk = np.fft.fft(per2)
    return ndk[ind_harm]

def analysisBodyProfileAr(signal, num=256, h=6):
    per2 = np.array([signal[round(i*(len(signal)-1)/(num-1))] for i in range(num)])
    feat = []
    for i in range(1,6):
        feat.append(per2[i*num//6-h: i*num//6+h].mean())
        
    return np.array(feat, dtype=complex)

def analysisBodyEdgeWavelet(signal, h=256):
    szc = signal.shape[0]
    L = pywt.dwt_max_level(szc,'sym4')
    coef = pywt.wavedec(signal, 'sym4', level=L)
    rcoef = coef_ravel(coef)
    plt.figure(figsize=(22,4))
    plt.plot(signal); plt.show()
    plt.figure(figsize=(22,4))
    plt.plot(rcoef[:512]); plt.show()
    return rcoef[:h]

def analysisBodyEdgeFourier(signal, h=128):
    szc = int(signal.shape[0]*0.45)
    signal = signal[:szc]*np.hanning(szc)
    signal = (signal-signal.mean())/signal.std()
    ind_harm = list(range(16,h+16)) + list(range(szc-h-16, szc-16))
    ndk = np.fft.fft(signal)
    return ndk[ind_harm]

def analysisApexFourier(signal, num=256, h=6):
    ind_harm = list(range(h//2)) + list(range(num-h//2, num))
    ix = signal.mean()
    per2 = [signal[round(i*(len(signal)-1)/(num-1))]-ix for i in range(num)]
    ndk = np.fft.fft(per2)
    return ndk[ind_harm]

def profileAnalysis(signal, num=256, h=6, c=0):
    ind_harm = list(range(h//2)) + list(range(num-h//2, num))
    ix = signal.mean()
    per2 = [signal[round(i*(len(signal)-1)/(num-1))]-ix for i in range(num)]
    ndk = np.fft.fft(per2)
    ndk[h//2:num-h//2] = 0
    #fig = plt.figure(figsize=(22,4))
    #plt.stem(abs(ndk)); plt.show()
    y1 = np.fft.ifft(ndk)
    #plt.plot(per2)
    plt.plot(real(y1), color=palette[c])
