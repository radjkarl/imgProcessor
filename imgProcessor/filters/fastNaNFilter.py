from __future__ import division

import numpy as np
from numba import jit

from skimage.transform import resize


def fastNaNFilter(arr, ksize=30,every=5, fn='median'):
    '''
    a fast 2d median filter for large kernel sizes that also 
    works with nans
    the computation speed is increased because only 'every'nsth position 
    within the median kernel is evaluated
    '''
    assert ksize > 3*every
    s0,s1 = arr.shape[:2]
    
    ss0 = s0//every
    every = s0//ss0
    ss1 = s1//every
    
    out = np.ones((ss0+1,ss1+1))
    ss0,ss1 = _calc(arr, out, ksize, every, typ=0 if fn=='median' else 1)
    out = out[:ss0,:ss1]
    return resize(out,arr.shape[:2])




@jit(nopython=True)
def _calc(arr, out, ksize,every, typ=0):
    gx = arr.shape[0]
    gy = arr.shape[1]
    ii = 0

    for i in range(0,gx,every):
        jj = 0
        for j in range(0,gy,every):
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
            #needs at least one not nan value, otherwise returns assertion error:
            sub = arr[xmn:xmx:every, ymn:ymx:every]
            if np.all(np.isnan(sub)):
                out[ii,jj]=np.nan
            else:
#             if np.argmax(np.isfinite(arr[xmn:xmx:every, ymn:ymx:every])):
                if typ == 0:
                    out[ii,jj]=np.nanmedian(sub)
                else:
                    out[ii,jj]=np.nanmean(sub)
#             else:
#                 out[ii,jj]=np.nan
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

    bg1 = fastNaNFilter(img,40,5, fn='median')
    bg2 = fastNaNFilter(img,40,5, fn='mean')
    
    if 'no_window' not in sys.argv:
        plt.figure('image')
        plt.imshow(img, interpolation='none')
        plt.colorbar()
        plt.figure('background - median')
        plt.imshow(bg1, interpolation='none')
        plt.colorbar()
        plt.figure('background - mean')
        plt.imshow(bg1, interpolation='none')
        plt.colorbar()
        plt.figure('difference (median)')
        plt.imshow(img-bg1, interpolation='none')
        plt.colorbar()

        
        plt.show()