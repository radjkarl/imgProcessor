
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
    ksize=3
    positions, _ = growPositions(ksize)
    print(positions)
    
    
    