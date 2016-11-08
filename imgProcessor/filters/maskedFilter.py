'''
Created on 1 Nov 2016

@author: elkb4
'''
from __future__ import division

import numpy as np
from numba import njit


def maskedFilter(arr, mask, ksize=30, fn='median'):
    '''
    fn['mean', 'median']
    replaced masked areas with filtered results
    '''
    if fn == 'median':
        raise Exception('[median] doesnt work at the moment')
    c = {#'median':_calcMedian,
         'mean':_calcMean,
         }[fn]
    c(arr, mask, ksize)
    return arr
    
#TODO: only filter method differs
# find better way for replace it than making n extra defs
@njit
def _calcMean(arr, mask, ksize):
    gx = arr.shape[0]
    gy = arr.shape[1]
    for i in range(gx):
        for j in range(gy):   
            #print (mask[i,j])
            if mask[i,j]:
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
                        if not mask[ii,jj]:
                            val += arr[ii,jj]
                            n += 1
                if n > 0: 
                    arr[i,j] = val/n
                


# 
# @njit
# def _calcMean(arr, mask, ksize):
#     gx = arr.shape[0]
#     gy = arr.shape[1]
#     for i in range(gx):
#         for j in range(gy):   
#             if mask[i,j]:
#                 xmn = i-ksize
#                 if xmn < 0:
#                     xmn = 0
#                 xmx = i+ksize
#                 if xmx > gx:
#                     xmx = gx 
#                 ymn = j-ksize
#                 if ymn < 0:
#                     ymn = 0
#                 ymx = j+ksize
#                 if ymx > gy:
#                     ymx = gy
#                 arr[i,i]=np.mean(arr[xmn:xmx, ymn:ymx])

   

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
    print (mask.sum())
    bg1 = maskedFilter(img,mask, 40, fn='mean')
    bg2 = maskedFilter(img,mask, 40, fn='mean')
    
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
        plt.figure('difference (median)')
        plt.imshow(img-bg1, interpolation='none')
        plt.colorbar()
        
        plt.show()