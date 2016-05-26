import numpy as np
from numba import jit


def closestDirectDistance(arr, ksize=30, dtype=np.uint16):
    '''
    return an array with contains the closest distance to the next positive
    value given in arr  within a given kernel size
    '''

    out = np.zeros_like(arr, dtype=dtype)
    _calc(out, arr,ksize)
    return out
    

@jit(nopython=True)
def _calc(out, arr, ksize):
    s0,s1 = arr.shape
    min_dist0 = 2*ksize
    for i in xrange(s0):
        for j in xrange(s1): 
            if arr[i,j]:
                out[i,j] = 0
            else:
                min_dist = min_dist0
                #for every pixel
                for ii in xrange(-ksize,ksize+1):
                    for jj in xrange(-ksize,ksize+1):
                        #find closest busbar within kernel size
                        if ii == 0 and jj == 0:
                            continue
                        xi = i +ii
                        yi = j +jj
                        #if within image
                        if 0 <= xi < s0  and 0 <= yi < s1:
                            if arr[xi,yi]:
                                dist = (ii**2 + jj**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                out[i,j] = min_dist



if __name__ == '__main__':
    import pylab as plt
    import sys
    
    size = 100,100
    arr = np.random.rand(*size) > 0.99
    
    out = closestDirectDistance(arr)

    if 'no_window' not in sys.argv:
        out[arr] = np.nan
        plt.imshow(out)
        plt.colorbar()
        plt.show()