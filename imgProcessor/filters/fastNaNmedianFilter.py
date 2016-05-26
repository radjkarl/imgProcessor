
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
    import sys
    import pylab as plt
    import imgProcessor
    from imgProcessor.imgIO import imread
    from fancytools.os.PathStr import PathStr
    
    p = PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence')
    
    img = imread(p.join('EL_module_orig.PNG'), 'gray')

    #add nans
    img[300:310]=np.nan
    img[:,110:130]=np.nan

    bg = fastNaNmedianFilter(img,40,5)
    
    if 'no_window' not in sys.argv:
        plt.figure('image')
        plt.imshow(img, interpolation='none')
        plt.figure('background')
        plt.imshow(bg, interpolation='none')
    
        plt.figure('difference')
        plt.imshow(img-bg, interpolation='none')
        
        plt.show()