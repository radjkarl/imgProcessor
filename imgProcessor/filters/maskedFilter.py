'''
Created on 1 Nov 2016

@author: elkb4
'''
from __future__ import division

import numpy as np
from numba import njit


def maskedFilter(arr, mask, ksize=30, fill_mask=True,
                 fn='median'):
    '''
    fn['mean', 'median']
    
    fill_mask=True:
        replaced masked areas with filtered results

    fill_mask=False:
    masked areas are ignored
    '''

    if fill_mask:
        mask1 = mask 
        out = arr
    else:
        mask1 = ~mask
        out = np.full_like(arr, fill_value=np.nan)
    mask2 = ~mask 
    
    if fn == 'mean':
        _calcMean(arr, mask1, mask2, out, ksize//2)
    else:
        buff = np.empty(shape=(ksize*ksize), dtype=arr.dtype)
        _calcMedian(arr, mask1, mask2, out, ksize//2, buff)
    return out
    
#TODO: only filter method differs
# find better way for replace it than making n extra defs
@njit
def _calcMean(arr, mask1, mask2, out, ksize):
    gx = arr.shape[0]
    gy = arr.shape[1]
    for i in range(gx):
        for j in range(gy):   
            #print (mask[i,j])
            if mask1[i,j]:
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
                val = 0
                n = 0
                for ii in range(xmn,xmx):
                    for jj in range(ymn,ymx):
                        if mask2[ii,jj]:
                            val += arr[ii,jj]
                            n += 1


                if n > 0: 
                    out[i,j] = val/n


@njit
def _calcMedian(arr, mask1, mask2, out, ksize, buff):
    gx = arr.shape[0]
    gy = arr.shape[1]
    for i in range(gx):
        for j in range(gy):   
            if mask1[i,j]:
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
                n=0
                for ii in range(xmn,xmx):
                    for jj in range(ymn,ymx):
                        if mask2[ii,jj]:
                            buff[n]=arr[ii,jj]
                            n += 1
                if n>0:
                    out[i,j] = np.median(buff[:n])

             

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

    mask = np.isnan(img)
    bg1 = maskedFilter(img.copy(),mask, 80, fn='mean')
    bg2 = maskedFilter(img.copy(),mask, 80, fn='median')
    
    if 'no_window' not in sys.argv:
        plt.figure('image')
        plt.imshow(img, interpolation='none')
        plt.colorbar()
        plt.figure('median')
        plt.imshow(bg1, interpolation='none')
        plt.colorbar()
        plt.figure('mean')
        plt.imshow(bg2, interpolation='none')
        plt.colorbar()
        plt.figure('difference (mean-median)')
        plt.imshow(bg1-bg2, interpolation='none')
        plt.colorbar()
        
        plt.show()