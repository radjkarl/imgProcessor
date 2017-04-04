# coding=utf-8
from __future__ import division

import numpy as np
from scipy.ndimage.filters import median_filter

from imgProcessor.imgIO import imread
from imgProcessor.camera.NoiseLevelFunction import oneImageNLF
from imgProcessor.utils.getBackground2 import getBackground2
from imgProcessor.transform.imgAverage import imgAverage
from imgProcessor.features.SingleTimeEffectDetection \
    import SingleTimeEffectDetection


def flatFieldFromCloseDistance(imgs, bg_imgs=None):
    '''
    Average multiple images of a homogeneous device
    imaged directly in front the camera lens.

    if [bg_imgs] are not given, background level is extracted
        from 1% of the cumulative intensity distribution
        of the averaged [imgs]

    This measurement method is referred as 'Method A' in
    ---
    K.Bedrich, M.Bokalic et al.:
    ELECTROLUMINESCENCE IMAGING OF PV DEVICES:
    ADVANCED FLAT FIELD CALIBRATION,2017
    ---
    '''
    img = imgAverage(imgs)
    bg = getBackground2(bg_imgs, img)
    img -= bg
    mx = median_filter(img[::10, ::10], 3).max()
    img /= mx
    return img


def flatFieldFromCloseDistance2(images, bgImages=None, calcStd=False,
                                nlf=None, nstd=6):
    '''
    Same as [flatFieldFromCloseDistance]. Differences are:
    ... single-time-effect removal included
    ... returns the standard deviation of the image average [calcStd=True]

    Optional:
    -----------
    calcStd -> set to True to also return the standard deviation
    nlf -> noise level function (callable)
    nstd -> artefact needs to deviate more than [nstd] to be removed
    '''

    if len(images) > 1:

        # start with brightest images
        def fn(img):
            img = imread(img)
            s0, s1 = img.shape[:2]
            # rough approx. of image brightness:
            return -img[::s0 // 10, ::s1 // 10].min()

        images = sorted(images, key=lambda i: fn(i))

        avgBg = getBackground2(bgImages, images[1])

        i0 = imread(images[0], dtype=float) - avgBg
        i1 = imread(images[1], dtype=float) - avgBg

        if nlf is None:
            nlf = oneImageNLF(i0, i1)[0]

        det = SingleTimeEffectDetection(
            (i0, i1), nlf, nStd=nstd, calcVariance=calcStd)

        for i in images[1:]:
            i = imread(i)
            # exclude erroneously darker areas:
            thresh = det.noSTE - nlf(det.noSTE) * nstd
            mask = i > thresh
            # filter STE:
            det.addImage(i, mask)

        ma = det.noSTE

    else:
        ma = imread(images[0], dtype=float) - avgBg

    # fast artifact free maximum:
    mx = median_filter(ma[::10, ::10], 3).max()

    if calcStd:
        return ma / mx, det.mma.var**0.5 / mx

    return ma / mx


if __name__ == '__main__':
    import sys
    from time import time
    from matplotlib import pyplot as plt
    from imgProcessor.equations.vignetting import vignetting

    # make 10 vignetting arrays with slightly different optical centre
    # to simulate effects that occur when vignetting is measured badly
    d = np.linspace(-20, 20, 100)
    bg = np.random.rand(100, 100) * 10
    vigs = [np.fromfunction(
            # vignetting from function
            lambda x, y:
            vignetting((x, y), cx=50 - di, cy=50 + di),
            (100, 100)) * 100 +
            # add noise
            np.random.rand(100, 100) * 10
            for di in d]
    ###
    t0 = time()
    avg = flatFieldFromCloseDistance(vigs, bg)
    t1 = time()
    print('[flatFieldFromCloseDistance] elapsed time: %s' % (t1 - t0))

    avg2, std2 = flatFieldFromCloseDistance2(vigs, bg, calcStd=True)
    t2 = time()
    print('[flatFieldFromCloseDistance2] elapsed time: %s' % (t2 - t1))
    ###
    if 'no_window' not in sys.argv:
        plt.figure('example vignetting img (1/%s)' % len(d))
        plt.imshow(vigs[0], interpolation='none')
        plt.colorbar()

        plt.figure('example vignetting img (10/%s)' % len(d))
        plt.imshow(vigs[-1], interpolation='none')
        plt.colorbar()

        plt.figure('averaged vignetting array')
        plt.imshow(avg, interpolation='none')
        plt.colorbar()

        plt.figure('averaged vignetting array [2]')
        plt.imshow(avg2, interpolation='none')
        plt.colorbar()

        plt.figure('standard deviation [2]')
        plt.imshow(std2)
        plt.colorbar()

        plt.show()
