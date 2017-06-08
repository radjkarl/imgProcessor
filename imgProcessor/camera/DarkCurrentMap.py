from __future__ import print_function

import numpy as np
from collections import OrderedDict

from fancytools.math.linRegressUsingMasked2dArrays \
    import linRegressUsingMasked2dArrays

from imgProcessor.imgIO import imread
from imgProcessor.features.SingleTimeEffectDetection \
    import SingleTimeEffectDetection
from imgProcessor.utils.baseClasses import Iteratives


# TODO: rename to STE free average??
class DarkCurrentMap(Iteratives):
    '''
    Averages given background images
    removing single time effects
    '''

    def __init__(self, twoImages, noise_level_function=None,
                 calcVariance=False, **kwargs):
        Iteratives.__init__(self, **kwargs)

        assert len(twoImages) > 1, 'need at least 2 images'

        self.det = SingleTimeEffectDetection(twoImages, noise_level_function,
                                             nStd=3, calcVariance=calcVariance)

    def addImg(self, img, raiseIfConvergence=False):
        self.det.addImage(img)
        if raiseIfConvergence:
            return self.checkConvergence(self.det.mma.var**0.5)

    def map(self):
        return self.det.mma.avg

    def uncertaintyMap(self):
        return self.det.mma.var**0.5

    def uncertainty(self):
        return np.mean(self.det.mma.var)**0.5


def averageSameExpTimes(imgs_path):
    '''
    average background images with same exposure time
    '''
    firsts = imgs_path[:2]
    imgs = imgs_path[2:]
    for n, i in enumerate(firsts):
        firsts[n] = np.asfarray(imread(i))
    d = DarkCurrentMap(firsts)
    for i in imgs:
        i = imread(i)
        d.addImg(i)
    return d.map()


def getLinearityFunction(expTimes, imgs, mxIntensity=65535, min_ascent=0.001,
                         ):
    '''
    returns offset, ascent 
    of image(expTime) = offset + ascent*expTime
    '''
    # TODO: calculate [min_ascent] from noise function
    # instead of having it as variable

    ascent, offset, error = linRegressUsingMasked2dArrays(
        expTimes, imgs, imgs > mxIntensity)

    ascent[np.isnan(ascent)] = 0
    # remove low frequent noise:
    if min_ascent > 0:
        i = ascent < min_ascent
        offset[i] += (0.5 * (np.min(expTimes) + np.max(expTimes))) * ascent[i]
        ascent[i] = 0

    return offset, ascent, error


def sortForSameExpTime(expTimes, img_paths):  # , excludeSingleImg=True):
    '''
    return image paths sorted for same exposure time
    '''
    d = {}
    for e, i in zip(expTimes, img_paths):
        if e not in d:
            d[e] = []
        d[e].append(i)
#     for key in list(d.keys()):
#         if len(d[key]) == 1:
#             print('have only one image of exposure time [%s]' % key)
#             print('--> exclude that one')
#             d.pop(key)
    d = OrderedDict(sorted(d.items()))
    return list(d.keys()), list(d.values())


def getDarkCurrentAverages(exposuretimes, imgs):
    '''
    return exposure times, image averages for each exposure time
    '''
    x, imgs_p = sortForSameExpTime(exposuretimes, imgs)
    s0, s1 = imgs[0].shape

    imgs = np.empty(shape=(len(x), s0, s1),
                    dtype=imgs[0].dtype)
    for i, ip in zip(imgs, imgs_p):
        if len(ip) == 1:
            i[:] = ip[0]
        else:
            i[:] = averageSameExpTimes(ip)
    return x, imgs


def getDarkCurrentFunction(exposuretimes, imgs, **kwargs):
    '''
    get dark current function from given images and exposure times
    '''
    exposuretimes, imgs = getDarkCurrentAverages(exposuretimes, imgs)
    offs, ascent, rmse = getLinearityFunction(exposuretimes, imgs, **kwargs)
    return offs, ascent, rmse


if __name__ == '__main__':
    import pylab as plt
    import sys

    # generate some random images for the following exposure times:
    exposuretimes = list(range(10, 100, 20)) * 3
    print('exposure times:, ', exposuretimes)
    offs = np.random.randint(0, 100, (30, 100))
    ascent = np.random.randint(0, 10, (30, 100))

    def noise(): return np.random.randint(0, 10, (30, 100))
    # calculate every image as function of exposure time
    # and add noise:
    imgs = [offs + t * ascent + noise() for t in exposuretimes]

    offs2, ascent2, rmse = getDarkCurrentFunction(exposuretimes, imgs)

    if 'no_window' not in sys.argv:
        plt.figure("image 1")
        plt.imshow(imgs[1])
        plt.colorbar()

        plt.figure("image 5")
        plt.imshow(imgs[5])
        plt.colorbar()

        plt.figure("calculated image 1")
        plt.imshow(offs2 + exposuretimes[1] * ascent2)
        plt.colorbar()

        plt.figure("calculated image 5")
        plt.imshow(offs2 + exposuretimes[5] * ascent2)
        plt.colorbar()

        plt.show()
