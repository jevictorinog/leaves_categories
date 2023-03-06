import matplotlib.pyplot as plt
from clusterTools import *

palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 
           'tab:cyan','mediumseagreen', 'coral', 'cornflowerblue', 'lawngreen', 'gold', 'burlywood','chocolate',
           'slateblue','slategray','orchid','tan','rosybrown', 'darkgoldenrod', 'silver', 'tab:gray', 'k']

def draw_points(points, categories, save=''):
    plt.figure(figsize=(18, 12))
    for i in range(len(categories)):
        plt.plot(points[i,0]+1, points[i,1]+1, '.', color=palette[categories[i]])
        plt.plot(points[i,2]+3, points[i,1]+1, '.', color=palette[categories[i]])
        plt.plot(points[i,0]+1, points[i,2]-1, '.', color=palette[categories[i]])
    plt.plot([0,1.9], [0,0], 'k--', [2,2],[2, 0.1], 'k--')
    plt.text(1, 2, "PC1 vs PC2", horizontalalignment='center', fontweight='bold')
    plt.text(3, 2, "PC2 vs PC3", horizontalalignment='center', fontweight='bold')
    plt.text(1,-1.9,"PC1 vs PC3", horizontalalignment='center', fontweight='bold')
    plt.axis('off')        
    if len(save) > 1: plt.savefig(save, format='svg')
    print('Representation space. samples:', len(categories))
    plt.show()

def draw_contours(contours, df, save=''):
    num_leaves = len(df.index)
    plt.figure(figsize =(14,num_leaves//2))
    for i in range(num_leaves//5*5): 
        plt.subplot(num_leaves//5, 5, i+1)
        plt.plot(contours[i,:,0], contours[i,:,1], color=palette[df.cat[i]])
        plt.text(0, 0, df.num[i], ha='center', color='white')
        plt.axis('equal')
        plt.axis('off')
    if len(save) > 1: plt.savefig(save, format='svg')
    plt.show()
    
def draw_ptsContours(points, contours, df, sc=0.08, save='', d1=0, d2=1):
    plt.figure(figsize=(20, 15))
    for i in range(len(df.index)):
        #if -0.1 < points[i,0] < 0.7 and -0.3 < points[i,1] < 0.2: 
            plt.plot(sc*contours[i,:,0]+points[i,d1], sc*contours[i,:,1]+points[i,d2], color=palette[df.cat[i]])
            plt.text(points[i,d1], points[i,d2], df.num[i], ha='center', color=palette[df.cat[i]])
    plt.axis('equal')
    plt.axis('off')
    if len(save) > 1: plt.savefig(save, format='svg')
    plt.show()
    print('Leaves distribution in the representation space. Projection each leaf samples in the representation space')    

def draw_ptsContoursInfo(points, contours, df, sc=0.12, save='', d1=0, d2=1):
    plt.figure(figsize=(20, 15))
    for i in range(len(df.index)):
        if i%20==0: 
            plt.plot(sc*contours[i,:,0]+points[i,d1], sc*contours[i,:,1]+points[i,d2], color=palette[df.cat[i]])
            plt.text(points[i,d1], points[i,d2], str(df.cat[i]+1), ha='center', color=palette[df.cat[i]])
        else:
            plt.plot(points[i,0], points[i,1], '.', color=palette[df.cat[i]])
    plt.axis('equal')
    plt.axis('off')
    tt = plt.show()

def draw_points5D(points, categories, save=''):
    fig, axs = plt.subplots(3,3, figsize=(24, 24))
    f = [0,2,4,0,3,4,0,1,4]
    for k in range(9):
        for i in range(points.shape[0]):
            x = f[k]; y = k//3+1;
            axs[k//3,k%3].plot(points[i,x], points[i,y], '.', color=palette[categories[i]])
            axs[k//3,k%3].set_title(f'PC{x} vs PC{y}'); axs[k//3,k%3].axis('off')

def draw_points1d(points, categories, I, N, dEdge, K, save='', d1=0, d2=1):
    plt.figure(figsize=(22, 12))
    for i in range(len(categories)):
        plt.plot(points[i,d1], points[i,d2], '.', color=palette[categories[i]])

    for k in range(K):
        i, j = get_index(I[k], N)
        if find_edge(dEdge, i, j, N):
            plt.plot([points[i,d1], points[j,d1]], [points[i,d2], points[j,d2]], color=palette[categories[i]])
    plt.axis('off')        
    if len(save) > 1: plt.savefig(save, format='svg')
    plt.show()
