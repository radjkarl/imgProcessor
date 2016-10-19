# coding=utf-8
from __future__ import division
from __future__ import print_function

import numpy as np

from scipy.ndimage.filters import maximum_filter, laplace


from fancytools.fit.fit2dArrayToFn import fit2dArrayToFn

from imgProcessor.interpolate.polyfit2d import polyfit2dGrid
from imgProcessor.equations.vignetting import vignetting, guessVignettingParam


def _highGrad(arr):
    # mask high gradient areas in given array
    s = min(arr.shape)
    return maximum_filter(np.abs(laplace(arr, mode='reflect')) > 0.01,  # 0.02
                          min(max(s // 5, 3), 15))


def function(img, mask, **kwargs):
    arr = fit2dArrayToFn(img, vignetting, mask=~mask,
                         guess=guessVignettingParam(img), **kwargs)[0]
    arr /= arr.max()
    return arr


def polynomial(img, mask, inplace=False, max_dev=1e-5, max_iter=20):
    '''
    replace all masked values
    calculate flatField from 2d-polynomal fit filling
    all high gradient areas within averaged fit-image

    returns flatField, average background level, fitted image, valid indices mask
    '''
    if inplace:
        out = img
    else:
        out = img.copy()
    lastm = 0
    for _ in range(max_iter):
        out2 = polyfit2dGrid(out, mask, order=2, copy=not inplace)

        res = (np.abs(out2 - out)).mean()
        print('residuum: ', res)
        if res < max_dev:
            out = out2
            break
        out = out2
        mask = _highGrad(out)

        m = mask.sum()
        if m == lastm or m == img.size:
            break
        lastm = m

    out = np.clip(out, 0.1, 1, out=out if inplace else None)
    return out
