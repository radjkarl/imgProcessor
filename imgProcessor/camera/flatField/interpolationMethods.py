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


def function(img, mask=None, replace_all=False, outgrid=None, fn=vignetting,
             guess=None, orthogonal=True, **kwargs):
    if outgrid is not None:
        replace_all = True
    if mask is None:
        mask = np.zeros_like(img, dtype=bool)
        replace_all = True
    down_scale_factor = 1
    if fn == vignetting and guess is None:
        s0, s1 = img.shape
        if ~mask.sum() > 1000:
            down_scale_factor = 0.3
        guess = guessVignettingParam(
            (s0 * down_scale_factor, s1 * down_scale_factor))
        if orthogonal:
            fn = lambda xy, f, alpha, cx, cy: vignetting(xy, f=f,
                                                         alpha=alpha,
                                                         cx=cx, cy=cy)
            f, alpha, rot, tilt, cx, cy = guess
            guess = (f, alpha, cx, cy)

    kwargs['down_scale_factor'] = down_scale_factor
    img2 = fit2dArrayToFn(img, fn,
                          mask=~mask, guess=guess,
                          outgrid=outgrid, **kwargs)[0]

    if not replace_all:
        img[mask] = img2[mask]
        img2 = img
    img2 /= img2.max()
    return img2


def polynomial(img, mask, inplace=False, replace_all=False,
               max_dev=1e-5, max_iter=20, order=2):
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
        out2 = polyfit2dGrid(out, mask, order=order, copy=not inplace,
                             replace_all=replace_all)
        if replace_all:
            out = out2
            break
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
    out = np.clip(out, 0, 1, out=out)  # if inplace else None)
    return out
