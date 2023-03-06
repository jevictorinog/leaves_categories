

def fourierAnalysis(contours, ):
    num_leaves = contours.shape[0]
    features   = np.zeros((250, num_harm), dtype=complex)
    j = (-1)**0.5
    for i in range(num_leaves):
        x   = contours[i,:,0] + contours[i,:,1]*j;
        dx  = np.diff(x, axis=0);
        ndx = np.divide(dx, np.absolute(dx))
        ndk = np.fft.fft(ndx.T)
        ndk = ndk.T

        features[i] = ndk[ind_harm]

    mfeat    = np.absolute(features[:num_leaves,:])
    pca      = PCA(n_components=min(num_component,num_leaves))
    pca.fit(mfeat)

    points = pca.transform(mfeat)
    pxdist = np.linalg.norm(points[:,0:3], axis=1)
    nd     = np.argsort(pxdist)
    print(f'clases:{sp}, files:{num_leaves}, file:{files[nd[0]]}')



