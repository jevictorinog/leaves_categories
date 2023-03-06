import cv2
import numpy as np
from sklearn.decomposition import PCA

def binImage(imName):
    im = cv2.imread(imName)
    if im is None: 
        print(f'Error binImage: problems loading file: {imname}')
        return 0
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, imBin = cv2.threshold(imgray, 127, 255, 0)
    return imBin


def getContour(imBin, reflex=True):
    cpoints, _ = cv2.findContours(imBin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not len(cpoints): 
        print(f'Error getContour: contours had not been found, size: {imBin.shape}')
        return 0
    edge = cpoints[0].ravel()
    elen = len(edge)
    contour = np.reshape(edge, (elen // 2, 2))

    xmean = np.mean(contour, axis=0)
    contour = contour - xmean
    xmax = np.max(contour)
    contour = contour / xmax
    if reflex: contour[:, [1]] = -1.0 * contour[:, [1]]

    return contour


def getContour(imBin, reflex=True):
    cpoints, _ = cv2.findContours(imBin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    edge = cpoints[0].ravel()
    elen = len(edge)
    contour = np.reshape(edge, (elen // 2, 2))

    xmean = np.mean(contour, axis=0)
    contour = contour - xmean
    xmax = np.max(contour)
    contour = contour / xmax
    if reflex: contour[:, [1]] = -1.0 * contour[:, [1]]

    return (contour)


def interparc(npoints, pxy):
    N = np.transpose(np.linspace(0, 1, npoints))
    nt = N.size

    # number of points on the curve
    n = np.size(pxy, 0)
    # pxy = np.array((pX,pY)).T
    p1 = pxy[0, :]
    pend = pxy[-1, :]
    last_segment = np.linalg.norm(np.subtract(p1, pend))
    epsilon = 10 * np.finfo(float).eps

    # If the two end points are not close enough lets close the curve
    if last_segment > epsilon * np.linalg.norm(np.amax(abs(pxy), axis=0)):
        pxy = np.vstack((pxy, p1))
        nt = nt + 1
    else:
        print('Contour already closed')

    pt = np.zeros((nt, 2))

    # Compute the chordal arclength of each segment.
    chordlen = (np.sum(np.diff(pxy, axis=0) ** 2.0, axis=1)) ** (1 / 2)
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength
    cumarc = np.append(0, np.cumsum(chordlen))

    tbins = np.digitize(N, cumarc)  # bin index in which each N is in

    # catch any problems at the ends
    tbins[np.where(tbins <= 0 | (N <= 0))] = 1
    tbins[np.where(tbins >= n | (N >= 1))] = n - 1

    s = np.divide((N - cumarc[tbins]), chordlen[tbins - 1])
    pt = pxy[tbins, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)

    return pt


def verticalStraightenAngle(imBin):
    [r, s] = np.where(imBin > 0)
    feat = np.array([r, s]).T
    pca = PCA(n_components=2)
    pca.fit(feat)
    pc1 = pca.components_[0, :]
    ang = -np.arctan(pc1[1] / pc1[0]) * 180 / np.pi
    return ang


def rotateBinImage(imBin, ang):
    h = imBin.shape[0]
    w = imBin.shape[1]
    rot = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=ang, scale=1)
    return cv2.warpAffine(imBin, rot, dsize=(w, h))


def rotatePoints(pts, ang):
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return np.matmul(pts, rot)


# -------------------------------------------------------------------------------

def sampling_leaves(df, amount_filter=0, percent=100, nsamples=0, lspecies=None, col='Species', seed=12):
    '''
    samplingLeaves: function to generate a partition of leaves dataset sampling to train purpose
    inputs:
      df: leaves dataframe wiht 'Species' column, image file reference and other attributes
      amount_filter: filter dataframe by amount of samples in each species
      col: optional, name of species column in dataframe
      percent: optional, determine the number of samples per specie to train or to test
      nsamples: optional, amount of samples to select per species
      lspecies: optional, list of species to select
      seed: optional, seed of random number generator
    outputs:
      ndx_train: array with index partition to train
      ndx_test: array with index partition to test
    '''
    np.random.seed(seed)
    ndx_train = np.array([], int)
    ndx_test = np.array([], int)
    lst_spec = df.groupby(col)[col].first().to_list()
    if lspecies:
        lst_spec = lspecies
    k = 0
    for estrato in lst_spec:
        dff = df[df[col] == estrato]
        if len(dff) >= amount_filter:
            ndx = dff.index.values
            ndx_perm = np.random.permutation(ndx)
            k = k + 1
            if nsamples > 0:
                n = nsamples if nsamples <= len(dff) else len(dff)
                ndx_train = np.hstack((ndx_train, ndx_perm[:n]))
                # print('%3d) %4d, 0 - [%4d]:  %s'%(k, n, len(dff), estrato))
            else:
                lim = int(len(ndx) * percent / 100 + 0.5)
                ndx_train = np.hstack((ndx_train, ndx_perm[:lim]))
                ndx_test = np.hstack((ndx_test, ndx_perm[lim:]))
                # print('%3d) %4d, %4d - [%4d]:  %s'%(k, lim, len(dff)-lim, len(dff), estrato))
    return ndx_train, ndx_test
