from __future__ import division

import numpy as np
from numba import njit
import cv2
from scipy.ndimage.filters import gaussian_filter


def fastFilter(arr, ksize=30, every=None, resize=True, fn='median',
                  interpolation=cv2.INTER_LANCZOS4,
                  smoothksize=0,
                  borderMode=cv2.BORDER_REFLECT):
    '''
    fn['nanmean', 'mean', 'nanmedian', 'median']
    
    a fast 2d filter for large kernel sizes that also 
    works with nans
    the computation speed is increased because only 'every'nsth position 
    within the median kernel is evaluated
    '''
    if every is None:
        every = max(ksize//3, 1)
    else:
        assert ksize >= 3*every
    s0,s1 = arr.shape[:2]
    
    ss0 = s0//every
    every = s0//ss0
    ss1 = s1//every
    
    out = np.full((ss0+1,ss1+1), np.nan)
    
    c = {'median':_calcMedian,
         'nanmedian':_calcNanMedian,
         'nanmean':_calcNanMean,
         'mean':_calcMean,
         }[fn]
    ss0,ss1 = c(arr, out, ksize, every)
    out = out[:ss0,:ss1]
    
    if smoothksize:
        out = gaussian_filter(out, smoothksize)
        
    
    if not resize:
        return out
    return cv2.resize(out, arr.shape[:2][::-1],
               interpolation=interpolation)


@njit
def _iter(arr, out, ksize,every):
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
            yield ii,jj, arr[xmn:xmx:every, ymn:ymx:every]
            jj += 1
        ii += 1


@njit
def _calcNanMedian(arr, out, ksize,every):
    for ii,jj, sub in _iter(arr, out, ksize,every):
        if np.all(np.isnan(sub)):
            out[ii,jj] = np.nan
        else:
            out[ii,jj]=np.nanmedian(sub)
#         else:
#         if np.all(np.isnan(sub)):
#             out[ii,jj]=np.nan
#         else:
#             out[ii,jj]=np.nanmedian(sub)
    return ii,jj


@njit
def _calcMean(arr, out, ksize,every, typ=0):
    for ii,jj, sub in _iter(arr, out, ksize,every):
        out[ii,jj]=np.mean(sub)
    return ii,jj


@njit
def _calcMedian(arr, out, ksize,every, typ=0):
    for ii,jj, sub in _iter(arr, out, ksize,every):
        out[ii,jj]=np.median(sub)
    return ii,jj


@njit
def _calcNanMean(arr, out, ksize,every, typ=0):
    for ii,jj, sub in _iter(arr, out, ksize,every):
        if np.all(np.isnan(sub)):
            out[ii,jj]=np.nan
        else:
            out[ii,jj]=np.nanmean(sub)
    return ii,jj
    
   
   

if __name__ == '__main__':
    import sys
    import pylab as plt
    import imgProcessor
    from imgProcessor.imgIO import imread
    from fancytools.os.PathStr import PathStr
    
    p = PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence')
    
    img = imread(p.join('EL_module_orig.PNG'), 'gray', dtype=float)

    #add nans
    img[300:320]=np.nan
    img[:,110:130]=np.nan

    bg1 = fastFilter(img,40,2, fn='nanmedian')
    bg2 = fastFilter(img,40,2, fn='nanmean')
    
    if 'no_window' not in sys.argv:
        plt.figure('image')
        plt.imshow(img, interpolation='none')
        plt.colorbar()
        plt.figure('background - nanmedian')
        plt.imshow(bg1, interpolation='none')
        plt.colorbar()
        plt.figure('background - nanmean')
        plt.imshow(bg2, interpolation='none')
        plt.colorbar()
        plt.figure('difference (median)')
        plt.imshow(img-bg1, interpolation='none')
        plt.colorbar()
        
        plt.show()