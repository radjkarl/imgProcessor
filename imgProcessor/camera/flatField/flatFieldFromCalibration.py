# coding=utf-8
from __future__ import division

import numpy as np
from scipy.ndimage.filters import median_filter

# from fancytools.math.MaskedMovingAverage import MaskedMovingAverage

from imgProcessor.camera.NoiseLevelFunction import oneImageNLF
from imgProcessor.imgIO import imread

from imgProcessor.utils.getBackground import getBackground
from imgProcessor.features.SingleTimeEffectDetection import SingleTimeEffectDetection
# from imgProcessor.utils.baseClasses import Iteratives


def flatFieldFromCalibration(images, bgImages=None, calcStd=False, nlf=None):
    '''
    returns a flat-field correction map
    through conditional average of multiple images reduced by a background image

    optional:
    calcStd -> set to True to also return the standard deviation
    nlf -> noise level function (callable)
    '''
    avgBg = getBackground(bgImages)

    if len(images) > 1:

        # start with brightest images
        def fn(img):
            img = imread(img)
            s0, s1 = img.shape[:2]
            # rough approx. of image brightness:
            return -img[::s0 // 10, ::s1 // 10].min()

        images = sorted(images, key=lambda i: fn(i))

        i0 = imread(images[0], dtype=float) - avgBg
        i1 = imread(images[1], dtype=float) - avgBg

        if nlf is None:
            nlf = oneImageNLF(i0, i1)[0]

        det = SingleTimeEffectDetection(
            (i0, i1), nlf, nStd=3, calcVariance=calcStd)

#         m = MaskedMovingAverage(shape=i0.shape, calcVariance=calcStd)
#         m.update(i0)

        for i in images[1:]:
            i = imread(i)
#             thresh = m.avg - nlf(m.avg) * 3

            # exclude erroneously darker areas:
            thresh = det.noSTE - nlf(det.noSTE) * 3
            mask = i > thresh

            # filter STE:
            det.addImage(i, mask)

#             m.update(i, ind)

        ma = det.noSTE

    else:
        ma = imread(images[0], dtype=float) - avgBg

    # fast artifact free maximum:
    mx = median_filter(ma[::10, ::10], 3).max()

    if calcStd:
        return ma / mx, det.mma.var**0.5 / mx

    return ma / mx


if __name__ == '__main__':
    from imgProcessor.equations.vignetting import vignetting
    from matplotlib import pyplot as plt
    import sys

    # make 10 vignetting arrays with slightly different optical centre
    # to simulate effects that occur when vignetting is measured badly
    d = np.linspace(-20, 20, 10)
    bg = np.random.rand(100, 100) * 10
    vigs = [np.fromfunction(lambda x, y:
                            vignetting((x, y), cx=50 - di, cy=50 + di),
                            (100, 100)) * 100 + bg for di in d]

    avg, std = flatFieldFromCalibration(vigs, bg, calcStd=True)

    if 'no_window' not in sys.argv:
        plt.figure('example vignetting img (1/10)')
        plt.imshow(vigs[0])
        plt.colorbar()

        plt.figure('example vignetting img (10/10)')
        plt.imshow(vigs[-1])
        plt.colorbar()

        plt.figure('averaged vignetting array')
        plt.imshow(avg)
        plt.colorbar()

        plt.figure('standard deviation')
        plt.imshow(std)
        plt.colorbar()

        plt.show()
