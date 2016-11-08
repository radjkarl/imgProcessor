# coding=utf-8
from __future__ import division
from __future__ import print_function

import numpy as np

from scipy.ndimage.filters import gaussian_filter, maximum_filter, \
    minimum_filter
from skimage.transform import rescale

from fancytools.math.MaskedMovingAverage import MaskedMovingAverage

from imgProcessor.imgIO import imread
from imgProcessor.measure.FitHistogramPeaks import FitHistogramPeaks
from imgProcessor.imgSignal import getSignalPeak, getSignalMinimum

from imgProcessor.utils.getBackground import getBackground


class FlatFieldFromImgFit(object):

    def __init__(self, images=None, bg_images=None,
                 #nstd=3, 
                 ksize=None, scale_factor=None):
        '''
        calculate flat field from multiple non-calibration images
        through ....
        * blurring each image
        * masked moving average of all images to even out individual deviations
        * fit vignetting function of average OR 2d-polynomal
        '''
        #self.nstd = nstd
        self.ksize = ksize
        self.scale_factor = scale_factor

        self.bglevel = 0  # average background level
        self._mx = 0
        self._n = 0
        self._m = None
        self._small_shape = None
        self._first = True

        self.bg = getBackground(bg_images)

        if images is not None:
            for n, i in enumerate(images):
                print('%s/%s' % (n + 1, len(images)))
                self.addImg(i)


    def _firstImg(self, img):

        if self.scale_factor is None:
            # determine so that smaller image size has 50 px
            self.scale_factor = 100 / min(img.shape)
        img = rescale(img, self.scale_factor)

        self._m = MaskedMovingAverage(shape=img.shape)
        if self.ksize is None:
            self.ksize = max(3, int(min(img.shape) / 10))
        self._first = False
        return img


    def _read(self, img):
        img = imread(img, 'gray', dtype=float)
        img -= self.bg
        return img


    @property
    def result(self):
        return self._m.avg


    def addImg(self, i):
        img = self._read(i)

        if self._first:
            img = self._firstImg(img)
        elif self.scale_factor != 1:
            img = rescale(img, self.scale_factor)
        try:
            f = FitHistogramPeaks(img)
        except AssertionError:
            return
        #sp = getSignalPeak(f.fitParams)
        mn = getSignalMinimum(f.fitParams)
        # non-backround indices:
        ind = img > mn#sp[1] - self.nstd * sp[2]
        # blur:
        blurred = minimum_filter(img, 3)#remove artefacts
        blurred = maximum_filter(blurred, self.ksize)
        gblurred = gaussian_filter(blurred, self.ksize)
        #blurred[ind] = gblurred[ind]

        # scale [0-1]:
        mn = img[~ind].mean()
        if np.isnan(mn):
            mn = 0
        mx = gblurred.max()
        gblurred -= mn
        gblurred /= (mx - mn)
        blurred -= mn
        blurred /= (mx - mn)
        
        ind = np.logical_and(ind, blurred > self._m.avg)

        self._m.update(gblurred, ind)
        self.bglevel += mn
        self._mx += mx

        self._n += 1


def vignettingFromDifferentObjects(imgs, bg):
    f = FlatFieldFromImgFit(imgs, bg)
    return f._m.avg, f._m.n > 0
