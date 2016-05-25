
import numpy as np
from numba import jit

from skimage.transform import resize


def fastNaNmedianFilter(arr, ksize=30,every=5):
    '''
    a fast 2d median filter for large kernel sizes that also 
    works with nans
    the computation speed is increased because only 'every'nsth position 
    within the median kernel is evaluated
    '''
    assert ksize > 3*every
    s0,s1 = arr.shape[:2]
    
    ss0 = int(round(s0/every))
    every = s0/ss0
    ss1 = s1/every
    
    out = np.ones((ss0+1,ss1+1))

    ss0,ss1 = _calc(arr, out, ksize, every)
    out = out[:ss0,:ss1]
    return resize(out,arr.shape[:2])



@jit(nopython=True)
def _calc(arr, out, ksize,every):
    gx = arr.shape[0]
    gy = arr.shape[1]
    ii = 0
    for i in xrange(0,gx,every):
        jj = 0
        for j in xrange(0,gy,every):
                      
            xmn = i-ksize
            if xmn < 0:
                xmn = 0
            xmx = i+ksize
            if xmx > gx:
                xmx = gx
                
            ymn = j-ksize
            if ymn < 0:
                ymn = 0
            ymx = j+ksize
            if ymx > gy:
                ymx = gy
            out[ii,jj]=np.nanmedian(arr[xmn:xmx:every, ymn:ymx:every])
            jj += 1
        ii += 1
    return ii,jj
   
   

if __name__ == '__main__':
    import pylab as plt
    s = (500,700)
    a = np.fromfunction(lambda x,y: 5*np.sin(0.1*x)+4*np.cos(0.01*y), s)
    #add noise
    a+= np.random.rand(*s)
    #add nans
    a[300:310]=np.nan
    a[:,300:400]=np.nan

    b = fastNaNmedianFilter(a,90,3)

    plt.figure('in')
    plt.imshow(a, interpolation='none')
    plt.figure('out')
    plt.imshow(b, interpolation='none')
    plt.show()