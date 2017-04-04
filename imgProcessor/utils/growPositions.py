
import numpy as np


def growPositions(ksize):
    '''
    return all positions around central point (0,0) 
    for a given kernel size 
    positions grow from smallest to biggest distances
    
    returns [positions] and [distances] from central cell
    
    '''
    i = ksize*2+1
    kk = np.ones( (i, i), dtype=bool)
    x,y = np.where(kk)
    pos = np.empty(shape=(i,i,2), dtype=int)
    pos[:,:,0]=x.reshape(i,i)-ksize
    pos[:,:,1]=y.reshape(i,i)-ksize

    dist = np.fromfunction(lambda x,y: ((x-ksize)**2
                                        +(y-ksize)**2)**0.5, (i,i))

    pos = np.dstack(
        np.unravel_index(
            np.argsort(dist.ravel()), (i, i)))[0,1:]

    pos0 = pos[:,0]
    pos1 = pos[:,1]

    return pos-ksize, dist[pos0, pos1]

if __name__ == '__main__':
    import pylab as plt
    import sys
    
    ksize=5
    positions, dist = growPositions(ksize)
    
    if 'no_window' not in sys.argv:
        px,py = positions[:,0], positions[:,1]
        
        plt.figure('positions')
        plt.plot(px,py)
        for n,(x, y) in enumerate(zip(px,py)):
            #number every marker:
            plt.text(x, y, str(n), color="red", fontsize=12)
        
        plt.figure('distances')
        plt.plot(dist)

        plt.show()

    
    
    