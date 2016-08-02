import numpy as np

from scipy.ndimage.filters import maximum_filter, laplace

from skimage.transform import resize

from fancytools.fit.fit2dArrayToFn import fit2dArrayToFn

from imgProcessor.interpolate.polyfit2d import polyfit2dGrid
from imgProcessor.equations.vignetting import vignetting, guessVignettingParam


def _highGrad(arr):
    #mask high gradient areas in given array 
    s = min(arr.shape)
    return maximum_filter(np.abs(laplace(arr, mode='reflect')) > 0.02,
                          min(max(s/5,3),15) )


def function(img, mask):
    arr = fit2dArrayToFn(img, vignetting, mask=~mask, 
                               guess=guessVignettingParam(img))[0]
    arr /= arr.max()            
    return arr


def polynomial(img, mask, inplace=False):
    
    '''
    calculate flatField from 2d-polynomal fit filling
    all high gradient areas within averaged fit-image
    
    returns flatField, average background level, fitted image, valid indices mask
    '''
    if inplace:
        out = img
    else:
        out = img.copy()
#     mask = ~mask
    lastm = 0
    for _ in xrange(10):
        out = polyfit2dGrid(out, mask, 2)
        mask =  _highGrad(out) 
        m = mask.sum()
        if m == lastm:
            break
        lastm = m

    out = np.clip(out,0.1,1) 
    return  out