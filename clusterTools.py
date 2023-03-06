import numpy  as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.spatial   import Delaunay
from scipy.optimize  import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

resultsTDA = np.zeros((2000, 4), dtype=int)
persistArrayH1 = []
persistArrayH2 = []

def get_index(Z, N):
    i = int((2*N - 1 - ((2*N-1)**2 - 8*Z)**0.5)/2)
    j = int(Z+1-i*(2*N-i-3)/2)
    return i, j

def iget_index(i, j, N):
    i, j = (j, i) if j<i else (i, j)
    return int(i*(2*N-i-3)/2 + j -1)

def get_alpha_dist(dist, arad):
    N = arad.size
    adist = np.zeros_like(dist)
    for k in range(N*(N-1)//2):
        i, j = get_index(k, N)
        adist[k] = dist[k] / (arad[i]+arad[j])
    return adist 

def alpha_radius_kde(X, kernel, bw):
    kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
    den = kde.score_samples(X)
    rad = 1/np.exp(den)
    return rad/rad.min()

def generate_edges(dTri, N):
    arr = []
    for i in range(dTri.simplices.shape[0]):
        ax = [0,0,0,1,1,2]
        bx = [1,2,3,2,3,3]
        for j in range(6):
            e1 = dTri.simplices[i,ax[j]]
            e2 = dTri.simplices[i,bx[j]]
            if e1 > e2: e1, e2 = e2, e1
            arr.append(e1*N+e2)
    arr = np.array(arr)
    dEdge = np.sort(arr)
    return np.unique(dEdge)

def find_edge(dEdge, a, b, N):
    if a > b: a,b = b,a
    val = a*N + b
    pos = np.searchsorted(dEdge, val)
    if pos >= dEdge.size: return False
    return val == dEdge[np.searchsorted(dEdge, val)]

def add_ev(i, k, ev, p1=-1, p2=-1):
    global resultsTDA
    T = resultsTDA.shape[0]
    if i >= T: 
        resultsTDA = np.resize(resultsTDA, (T*2, 4))
        print('RESIZE resultsTDA:', resultsTDA.shape)
    resultsTDA[i,0] = k;  resultsTDA[i, 1] = ev
    resultsTDA[i,2] = p1; resultsTDA[i, 3] = p2
    if i%100 == 0: print(i, '\t', resultsTDA[i,:])

def persist_h1(H1, N, k, Q=8):
    global persistArrayH1
    u1, c1 = np.unique(H1[H1!=N], return_counts=True)
    persistArrayH1.append([k, len(u1[c1>=Q])])
    
def persist_h2(H2, N, k, Q=8):
    global persistArrayH2
    u2, c2 = np.unique(H2[H2!=N], return_counts=True)
    persistArrayH2.append([k, len(u2[c2>=Q])])

def meanshift_results2(I, N, dEdge, Q=8):
    k, e, ng1, ng2 = 0, 0, 0, 0
    H1 = np.ones(N, dtype=np.int32)*N
    H2 = np.ones(N, dtype=np.int32)*N

    sim, ltri = [], []

    for i in range(N): sim.append(set())

    while H2[0]>=N or sum(H2==H2[0])<N:
        i, j = get_index(I[k], N)
        if find_edge(dEdge, i, j, N):
            if H1[i] == H1[j] == N:           # nuevo grupo
                H1[i] = H1[j] = ng1
                add_ev(e, k, 0, ng1); e+=1
                ng1 = ng1 + 1
                persist_h1(H1, N, k, Q)
            elif H1[i] == N: H1[i] = H1[j]; add_ev(e, k, 1, H1[j]); e+=1    # j se une al grupo de i
            elif H1[j] == N: H1[j] = H1[i]; add_ev(e, k, 1, H1[i]); e+=1   # i se une al grupo de j
            elif H1[i] != H1[j]:              # Fusion de grupos
                a = min(H1[i], H1[j])
                b = max(H1[i], H1[j])
                H1[H1==b] = a
                add_ev(e, k, 2, a, b); e+=1
                persist_h1(H1, N, k, Q)
            sim[i].add(j)
            sim[j].add(i)
            tri = sim[i].intersection(sim[j])
            for t in range(len(tri)):
                q  = tri.pop()
                nx = np.array([i, j, q])
                ltri.append(nx)
                nogr = np.count_nonzero(H2[nx]==N)
                if nogr == 3: # nuevo grupo
                    H2[nx] = ng2
                    add_ev(e, k, 3, ng2, q); e+=1
                    ng2    = ng2 + 1
                    persist_h2(H2, N, k, Q)
                elif nogr == 2:
                    a = H2[nx].min()
                    H2[nx] = a
                    add_ev(e, k, 4, a, q); e+=1
                else: # nogr < 2
                    if H2[i] == H2[j] == H2[q]: continue
                    a = H2[nx].min()
                    if H2[i] < N and H2[i] != a: H2[H2==H2[i]] = a
                    if H2[j] < N and H2[j] != a: H2[H2==H2[j]] = a
                    if H2[q] < N and H2[q] != a: H2[H2==H2[q]] = a
                    H2[nx] = a
                    add_ev(e, k, 5, a, q); e+=1
                    persist_h2(H2, N, k, Q)
        k = k+1
        if k >= I.size: break
    print('Final it =', e, 'k =', k)
    return H1, H2, ltri, e

def groupsH2(dfr, N, umbral):
    H2 = np.ones(N, dtype=np.int32)*N
    for i in range(umbral):
        a = dfr.gr1.iloc[i] 
        if dfr.event.iloc[i] == 3 or dfr.event.iloc[i] == 4:
            H2[dfr.i_.iloc[i]]  = a
            H2[dfr.j_.iloc[i]]  = a
            H2[dfr.gr2.iloc[i]] = a
        elif dfr.event.iloc[i] == 5:
            if H2[dfr.i_.iloc[i]]  < N: H2[H2==H2[dfr.i_.iloc[i]]]  = a
            if H2[dfr.j_.iloc[i]]  < N: H2[H2==H2[dfr.j_.iloc[i]]]  = a
            if H2[dfr.gr2.iloc[i]] < N: H2[H2==H2[dfr.gr2.iloc[i]]] = a
            H2[dfr.i_.iloc[i]]  = a
            H2[dfr.j_.iloc[i]]  = a
            H2[dfr.gr2.iloc[i]] = a
    u2, c2 = np.unique(H2, return_counts=True)
    H2p = [N if i in u2[c2<5] else i for i in H2]
    return H2p


def cost(m):
    s = np.max(m)
    return (- m + s)

def evaluation(clusters, classLeaf, species):
    num_leaves  = len(clusters)
    num_species = len(np.unique(classLeaf))

    set_cluster, ndx_cluster = np.unique(clusters, return_inverse=True)
    szc       = len(set_cluster)
    set_class = np.array(range(max(num_species, szc)))
    y_tmp     = set_class[ndx_cluster]
    mconf     = np.array(confusion_matrix(classLeaf, y_tmp))
    rr, ch    = linear_sum_assignment(cost(mconf))
    set_cross = np.argsort(ch)
    predict   = set_cross[ndx_cluster]
    score     = f1_score(classLeaf, predict, average='micro')
    col_names = [set_cluster[x] if x < szc else -(ch[x]+1) for x in ch]
    df        = pd.DataFrame(mconf[:num_species,ch], columns=col_names)
    df['species'] = species
    col_names.insert(0,'species')
    df_confus = df.reindex(columns=col_names)
    df_confus['Total'] = df_confus.sum(axis=1)

    print('FINAL F1 SCORE :', score)
    print('--'*50, '\nCONFUSION MATRIX:')
    display(df_confus)
    return df_confus

