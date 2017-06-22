# coding=utf-8
from __future__ import division

import numpy as np
from fancytools.fit.fit2dArrayToFn import fit2dArrayToFn
from imgProcessor.equations.vignetting import vignetting, guessVignettingParam


def function(img, mask=None, replace_all=False, outgrid=None, fn=vignetting,
             guess=None, orthogonal=True, **kwargs):
    """
    Functional approach (Kang-Weiss equation by default)
    for post processing of measured flat field images.
    """
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
            def fn(xy, f, alpha, cx, cy): return vignetting(xy, f=f,
                                                            alpha=alpha,
                                                            cx=cx, cy=cy)
            f, alpha, _rot, _tilt, cx, cy = guess
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
