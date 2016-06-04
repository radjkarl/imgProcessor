import numpy as np
from numba import jit


def edgesFromBoolImg(arr):
    '''
    takes a binary image (usually a mask) 
    and returns the edges of the object inside
    '''
    out = np.zeros_like(arr)
    _calc(arr, out)
    _calc(arr.T, out.T)
    return out


@jit(nopython=True)
def _calc(arr, out):
    gx = arr.shape[0]
    gy = arr.shape[1]

    for i in xrange(gx):
        last_val = arr[i,0]
        for j in xrange(1, gy):
            val = arr[i,j]
            if val != last_val:
                if val == 0:
                    #have edge within arr==True
                    j -=1
                out[i,j]=1
                last_val = val

if __name__ == '__main__':
    import pylab as plt
    import sys
    a = np.zeros((10,10), dtype=bool)
    a[:,5:]=True
    a[5:,:]=True
    b = edgesFromBoolImg(a)
    
    if 'no_window' not in sys.argv:
        plt.figure('in')
        plt.imshow(a, interpolation='none')
        plt.figure('out')
        plt.imshow(b, interpolation='none')
        plt.show()