import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 
           'tab:cyan','mediumseagreen', 'coral', 'cornflowerblue', 'lawngreen', 'gold', 'burlywood','chocolate',
           'slateblue','slategray','orchid','tan','rosybrown', 'darkgoldenrod', 'silver', 'tab:gray', 'k']

def persistencePlot(x1, y1, x2, y2, save=''):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.01, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=x2, y=y2, name=''  , line_shape='hv', marker=dict(color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x1, y=y1, name='H1', line_shape='hv', marker=dict(color='royalblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x1, y=y1, name=''  , line_shape='hv', marker=dict(color='lightgray')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x2, y=y2, name='H2', line_shape='hv', marker=dict(color='coral')), row=2, col=1)
    fig.update_layout(height=760, width=960, title_text='Persistence of groups', plot_bgcolor='white', yaxis=dict(title='Filtration value'))
    fig.show()
    if len(save) > 1: fig.write_image(save)

def groupsPersistPlot(x1, y1, x2, y2, save=''):
    grs = np.ediff1d(y1, to_begin=0) != 0
    val = y1[grs]
    rad = x1[grs]*100
    drad = np.ediff1d(rad, to_end=rad[-1]-rad[-2])
    tk   = np.arange(1, max(y1)+1)
    pers = np.array([max(drad[val==tk[i]]) for i in range(tk.size)])

    grs = np.ediff1d(y2, to_begin=0) != 0
    val = y2[grs]
    rad = x2[grs]*100
    drad = np.ediff1d(rad, to_end=rad[-1]-rad[-2])
    tk2  = np.arange(1, max(y2)+1)
    per2 = np.array([max(drad[val==tk2[i]]) for i in range(tk2.size)])

    dfp = pd.DataFrame({'gr':np.concatenate((tk,tk2)), 'pers':np.concatenate((pers,per2)).round(2)})
    dfp['kind'] = ['H1']*tk.size + ['H2']*tk2.size 
    fig = px.bar(dfp[(dfp.gr<10)&(dfp.gr>1)], x="gr", y="pers", text_auto=True, color='kind', barmode='group', width=1200, title='Persistence of groups', labels={'pers':'Persistence', 'gr':'Number of groups'})
    fig.update_layout(plot_bgcolor='white', bargroupgap=0.2, height=400, xaxis=dict(tickvals=tk))
    fig.data[0].marker.color = 'skyblue'
    fig.data[1].marker.color = '#FFBB66'
    fig.show()
    if len(save) > 1: fig.write_image(save)


def life_groupsH2(dfr, N, Q=8):
    # life_groupsH2 function that calculates the group life cycle: birth, dead, Qdead, fusion, duration
    H1 = np.ones(N, dtype=np.int32)*N
    H2 = np.ones(N, dtype=np.int32)*N
    G = dfr[dfr.event>2].gr1.max()+1
    gtime = np.zeros(G)
    gdead = np.zeros(G)
    ngr = []
    fusion = np.zeros(G, dtype=np.int32)
    elm = np.zeros(G, dtype=np.int32)
    sam  = [0]*G
    for i in range(dfr.shape[0]):
        a = dfr.gr1.iloc[i] 
        if dfr.event.iloc[i] == 3 or dfr.event.iloc[i] == 4:
            H2[dfr.i_.iloc[i]]  = a
            H2[dfr.j_.iloc[i]]  = a
            H2[dfr.gr2.iloc[i]] = a
            val = np.count_nonzero(H2==a)
            if val >= Q and gtime[a] == 0:
                gtime[a] = dfr.rend.iloc[i]
                ngr.append(a)
        elif dfr.event.iloc[i] == 5:
            b = H2[dfr.i_.iloc[i]]
            c = H2[dfr.j_.iloc[i]]
            d = H2[dfr.gr2.iloc[i]]
            tm =  dfr.rend.iloc[i]
            if b<N and b != a:             gdead[b] = tm; sam[b] = np.where(H2==b)[0]; fusion[b]=a; H2[H2==b] = a
            if c<N and c not in [a, b]:    gdead[c] = tm; sam[c] = np.where(H2==c)[0]; fusion[c]=a; H2[H2==c] = a
            if d<N and d not in [a, b, c]: gdead[d] = tm; sam[d] = np.where(H2==d)[0]; fusion[d]=a; H2[H2==d] = a

            H2[dfr.i_.iloc[i]] = H2[dfr.j_.iloc[i]] = H2[dfr.gr2.iloc[i]] = a
            val = np.count_nonzero(H2==a)
            if val >= Q and gtime[a] == 0:
                gtime[a] = dfr.rend.iloc[i]
                ngr.append(a)
                #print(f'({i})\tNuevo \t\tGR_{a}[{val}], \ttime: {round(gtime[a],3)}')
            if b<N and b != a and 0 < gtime[b] < gtime[a]:           gtime[a] = gtime[b] 
            if c<N and c not in [a,b] and 0 < gtime[c] < gtime[a]:   gtime[a] = gtime[c] 
            if d<N and d not in [a,b,c] and 0 < gtime[d] < gtime[a]: gtime[a] = gtime[d] 
        u, w = np.unique(H2[H2<N], return_counts=True)
        elm[u] = w
    gdead[np.argmin(gdead)] =  dfr.rend.max()
    dfx = pd.DataFrame(zip(ngr, gtime[ngr], gdead[ngr], elm[ngr], fusion[ngr]), columns=['group','birth', 'dead', 'qleaves', 'fusion'])
    dfx['duration'] = dfx.dead-dfx.birth
    if sam[0] == 0: sam[0] = np.arange(N)
    return dfx, sam 

def disagregate_leaves(dfx, sam, num_regs, sw):
    if sw: return 
    #df1 = dfx.sort_values('duration', ascending=False)
    df1 = dfx.sort_values(['duration', 'group'], ascending=[False,True])
    arr = list(df1.qleaves)
    grp = list(df1.group)
    for i in range(1, num_regs):
        n = grp.index(df1.fusion.iloc[i])
        arr[n] -= df1.qleaves.iloc[i]
        sam[df1.fusion.iloc[i]] = np.setdiff1d(sam[df1.fusion.iloc[i]], sam[grp[i]])
    arr = pd.Series(arr, index=df1.index, dtype=np.int32)
    return arr, True

def draw_treeCluster(dfx, num_regs=12, save=''):
    df1 = dfx.sort_values(['duration', 'group'], ascending=[False,True]).head(num_regs)
    df2 = df1.sort_values('dead')
    bars = []
    ndx = []
    bars.append(df1.group.iloc[0])
    ndx.append(df1.index[0])
    for i in range(1, num_regs):
        n = bars.index(df1.fusion.iloc[i])
        bars.insert(n+1, df1.group.iloc[i])
        ndx.insert(n+1, df1.index[i])

    pos = list(range(num_regs))
    ini = [dfx.birth.iloc[i] for i in ndx]
    fig, ax = plt.subplots(figsize=(20, 10))
    lea = df1.qleaves

    for i in range(num_regs):
        g = df2.group.iloc[i]
        f = df2.fusion.iloc[i]
        g1 = bars.index(g)
        f1 = bars.index(f)
        i1 = ndx[g1]
        i2 = ndx[f1]
        p1x = p2x = pos[f1]
        p3x = p4x = pos[g1]
        p1y = ini[f1]
        p4y = ini[g1]
        p2y = p3y = dfx.dead.iloc[i1]
        w1 = (lea[i1]/dfx.qleaves.sum())**0.5*8
        w2 = (lea[i2]/dfx.qleaves.sum())**0.5*8
        ds = 0.025*dfx.dead.max()
        #print(g, '-->', f, 'ndx', i2, i1, '\tx:', p1x, p3x, '\ty', round(p1y,3), round(p4y,3), round(p2y,3), '\tlea:', lea[i1], lea[i2])
        plt.plot([p1x, p2x], [p1y, p2y], lw=w2, color=palette[f1])
        plt.plot([p2x, p3x], [p2y, p3y], lw=w2, color=palette[f1])
        plt.plot([p3x, p4x], [p3y, p4y], lw=w1, color=palette[g1])
        plt.text(pos[g1]+0.05, (p4y+p3y)/2, round(dfx.duration.iloc[i1],3), color=palette[g1], rotation='vertical', va='center')
        plt.text(pos[g1], ini[g1]-ds, lea[i1], color=palette[g1], ha='center')
        plt.text(pos[f1], ini[f1]-ds, lea[i2], color=palette[f1], ha='center')
        pos[f1] = (pos[g1]+pos[f1])/2
        ini[f1] = p2y
        lea[i1] = lea[i2] = lea[i1]+lea[i2]
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(bars)
    if len(save) > 1: plt.savefig(save, format='svg')
    return bars

def draw_tree_leaf(points, contours, bars, sam, save=''):
    fig, ax = plt.subplots(figsize=(20, 3))
    for i, b in enumerate(bars):
        cn = points[sam[b]].mean(axis=0)
        ds = ((points[sam[b]] - cn)**2).sum(axis=1)
        n = np.argmin(ds)
        nc = sam[b][n]
        plt.plot(contours[nc,:,0]+i*1.5, contours[nc, :, 1], color=palette[i]); 
        plt.text(i*1.5, 0, str(sam[b].size), ha='center', color=palette[i])
    tt=plt.axis('equal')
    if len(save) > 1: plt.savefig(save, format='svg')

def interchange(a, vals1, vals2):
    m = [a==x for x in vals1]
    for i in range(len(m)):
        a[m[i]] = vals2[i]
    return a
