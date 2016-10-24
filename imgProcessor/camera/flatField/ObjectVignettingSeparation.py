# coding=utf-8
from __future__ import division
from __future__ import print_function

import numpy as np

import cv2

from scipy.ndimage.filters import maximum_filter, minimum_filter  # , median_filter
#from scipy.ndimage import gaussian_filter


from fancytools.math.boundingBox import boundingBox
from fancytools.math.MaskedMovingAverage import MaskedMovingAverage

from imgProcessor.imgIO import imread
from imgProcessor.utils.decompHomography import decompHomography
from imgProcessor.features.PatternRecognition import PatternRecognition


from imgProcessor.filters.fastNaNFilter import fastNaNFilter


# START VEREINFACHEN
# FlatFieldFromImgFit rename into SubImgFlatField
# FlatFieldFromImgFit rename into FullImgFlatField
# NEIN: dieses von FFbase class erben lassen welche flatFieldFromFit hat

class ObjectVignettingSeparation(PatternRecognition):
    """
    If an imaged object is superimposed by a flat field map
    (often determined by vignetting) the actual object signal can
    be separated from the cameras flatField using multiple imaged of the object
    at different positions. For this the following steps are needed:
    1. Set the first given image as reference.
   For every other image ... ( .addImg() )
    2. Calculate translation, rotation, shear - difference through pattern recognition
    3. Warp every image in order to fit the reference one.
    4. Set an initial flatField image from the local maximum of every image
   Iterate:
    5. Divide every warped image by its flatField.
    6. Define .object as the average of all fitted and flatField corrected images
    7. Extract .flatField as the ratio of (fitted) .object to every given image

    Usage:

    >>> o = ObjectFlatFieldSeparation(ref_img)
    >>> for img in imgs:
    >>>     o.addImg(img)
    >>> flatField, obj = o.separate()
    """

    def __init__(self, img, bg=None, maxDev=1e-4, maxIter=10):
        """
        Args:
            img (path or array): Reference image
        Kwargs:
            bg (path or array): background image - same for all given images
            maxDev (float): Relative deviation between the last two iteration steps
                            Stop iterative refinement, if deviation is smaller
            maxIter (int): Stop iterative refinement after maxIter steps
        """
        self.maxDev = maxDev
        self.maxIter = maxIter

        img = imread(img, 'gray')

        self.bg = bg
        if bg is not None:
            self.bg = imread(bg, 'gray', dtype=img.dtype)
            img = cv2.subtract(img, self.bg)

        # CREATE TEMPLATE FOR PATTERN COMPARISON:
        pos = self._findObject(img)
        self.obj_shape = img[pos].shape

#         plt.imshow(img)
#         print pos
#         plt.show()

        PatternRecognition.__init__(self, img[pos])

#         self.flatField = np.zeros(shape=img.shape, dtype=np.float64)
        self._ff_mma = MaskedMovingAverage(shape=img.shape,
                                           dtype=np.float64)

        self.object = None

        self.Hs = []    # Homography matrices of all fitted images
        self.Hinvs = []  # same, but inverse
        self.fits = []  # all imaged, fitted to reference

        self._refined = False

    # TODO: remove that property?
    @property
    def flatField(self):
        return self._ff_mma.avg

    @flatField.setter
    def flatField(self, arr):
        self._ff_mma.avg = arr

    def addImg(self, img, maxShear=0.015, maxRot=100, minMatches=12,
               borderWidth=100):
        """
        Args:
            img (path or array): image containing the same object as in the reference image
        Kwargs:
            maxShear (float): In order to define a good fit, refect higher shear values between
                              this and the reference image
            maxRot (float): Same for rotation
            minMatches (int): Minimum of mating points found in both, this and the reference image
        """
        try:
            fit, img, H, H_inv, nmatched = self._fitImg(img)
        except Exception as e:
            print(e)
            return

        # CHECK WHETHER FIT IS GOOD ENOUGH:
        (translation, rotation, scale, shear) = decompHomography(H)
        print('Homography ...\n\ttranslation: %s\n\trotation: %s\n\tscale: %s\n\tshear: %s'
              % (translation, rotation, scale, shear))
        if (nmatched > minMatches
                and abs(shear) < maxShear
                and abs(rotation) < maxRot):
            print('==> img added')
            # HOMOGRAPHY:
            self.Hs.append(H)
            # INVERSE HOMOGRSAPHY
            self.Hinvs.append(H_inv)
            # IMAGES WARPED TO THE BASE IMAGE
            self.fits.append(fit)

            # ADD IMAGE TO THE INITIAL flatField ARRAY:
#             smooth = gaussian_filter(img, sigma=5)
            i = img > self.signal_ranges[-1][0]
            # remove borders (that might have erroneous light):
            i = minimum_filter(i, borderWidth)
            self._ff_mma.update(img, i)

#
#             img[~i] = 0
#             ii = self.flatField < img
#             self.flatField[ii] = img[ii]
#             import pylab as plt
#             plt.imshow(self.flatField)
#             plt.figure(2)
#             plt.imshow(i)
#             print self.signal_ranges
#             plt.show()
            # image added
            return True
        return False

    def separate(self):
        self._createInitialflatField()
        for step in self:
            print('iteration step %s/%s' % (step, self.maxIter))

#         import pylab as plt
#         plt.figure(1)
#         plt.imshow(self.flatField)
#         plt.colorbar()
#         plt.show()

#         return self.flatField,self.flatField<0.3
        smoothed_ff, mask = self.smooth()
        return smoothed_ff, mask, self.flatField, self.object

    def smooth(self):
        #         ff = self.flatField.copy()
        #         ff[ff<0.3]= np.nan

        # TODO: there is non nan in the ff img, or?
        mask = self.flatField == 0
        from skimage.filters.rank import median, mean
        from skimage.morphology import disk

        ff = mean(median(self.flatField, disk(5), mask=~mask),
                  disk(13), mask=~mask)
#         mask = np.isfinite(ff)
#         import pylab as plt
#         plt.figure(1)
#         plt.imshow(ff)
#         plt.figure(2)
#         plt.imshow(self.flatField)
#         plt.figure(3)
#         plt.imshow(ff==0)
#         plt.show()
        return ff.astype(float) / ff.max(), mask

    def __iter__(self):
        # use iteration to refine the flatField array

        remove_border_size = 20
        feature_size = 50

        # keep track of deviation between two iteration steps
        # for break criterion:
        self._last_dev = None
        self.n = 1  # iteration number

        # CREATE IMAGE FIT MASK:
        self._fit_masks = []
        for i, f in enumerate(self.fits):
                # INDEX FOREGROUND:
            mask = f < self.signal_ranges[i][0]
            mask = maximum_filter(mask, feature_size)
#             plt.imshow(self.flatField)
# #             plt.figure(2)
# #             plt.imshow(f)
# #             from imgProcessor.signal import signalRange
# #
# #             print self.signal_ranges[i], signalRange(f)
#             plt.show()
            # IGNORE BORDER
            if remove_border_size:
                mask[:remove_border_size, :] = 1
                mask[-remove_border_size:, :] = 1
                mask[:, -remove_border_size:] = 1
                mask[:, :remove_border_size] = 1
            self._fit_masks.append(mask)

        return self

    def __next__(self):
        # THE IMAGED OBJECT WILL BE AVERAGED FROM ALL
        # INDIVITUAL IMAGES SHOWING THIS OBJECT FROM DIFFERENT POSITIONS:
        obj = MaskedMovingAverage(shape=self.obj_shape)
        for f, h in zip(self.fits, self.Hinvs):
            warpedflatField = cv2.warpPerspective(self.flatField,
                                                  h, (f.shape[1], f.shape[0]))
            obj.update(f / warpedflatField, warpedflatField != 0)
        self.object = obj.avg

        # THE NEW flatField WILL BE OBTAINED FROM THE WARPED DIVIDENT
        # BETWEEN ALL IMAGES THE THE ESTIMATED IMAGE OOBJECT
        sh = self.flatField.shape
        s = MaskedMovingAverage(shape=sh)
        #mx = 0
        for f, mask, h in zip(self.fits, self._fit_masks, self.Hs):
            div = f / self.object
            # ->do not interpolate between background and image border
            div[mask] = np.nan
            div = cv2.warpPerspective(div, h, (sh[1], sh[0]),  # borderMode=cv2.BORDER_TRANSPARENT
                                      )
            div = np.nan_to_num(div)
            s.update(div, div != 0)


#         plt.imshow(self.flatField)
#         plt.figure(2)
#         plt.imshow(s.avg)
#         plt.figure(3)
#         plt.imshow(f)
#
# #         print self.signal_ranges
#         plt.show()

        new_flatField = s.avg

#         every = int(100/3.5)
#         new_flatField = fastNaNFilter(new_flatField, 100, every)

        # STOP ITERATION?
        # RMSE excluding NaNs:
        dev = np.nanmean((new_flatField[::10, ::10]
                          - self.flatField[::10, ::10])**2)**0.5
        print('residuum: %s' % dev)
        if self.n > self.maxIter or (self._last_dev and (
            (self.n > 4 and dev > self._last_dev)
                or dev < self.maxDev)):
            #             self.flatField_mask = np.logical_or(s.n==0,s.avg<0)
            #             self.flatField_mask = np.logical_or(s.n==0,)
            raise StopIteration

        # remove erroneous values:
        self.flatField = np.clip(new_flatField, 0, 1)

        self.n += 1
        self._last_dev = dev
        return self.n

    def _createInitialflatField(self, downscale_size=9
                                # gausian_filter_size=None
                                ):

        #         if gausian_filter_size is None:
        #             gausian_filter_size = min(self.flatField.shape)/5

        # initial array
        s0, s1 = self.flatField.shape
#         sa = downscale_size
#         sb = int(round(sa * float(min(s0,s1))/max(s0,s1)))
#         if s0>s1:
#             ss0,ss1 = sa,sb
#         else:
#             ss0,ss1 = sb,sa

#         coarse = coarseMaximum(median_filter(self.flatField,3), (ss0,ss1))
#
#         s = self.flatField = resize( coarse,#resize(self.flatField, (ss0,ss1)),
#                                     (s0,s1), order=3 )

        f = int(max(s0, s1) / downscale_size)
        every = int(f / 3.5)

        s = self.flatField = fastNaNFilter(self.flatField, f, every)

#         plt.imshow(self.flatField)
#         plt.colorbar()
#         plt.show()

#         s = self.flatField = gaussian_filter(
#                                 maximum_filter(self.flatField,
#                                     size=gausian_filter_size),
#                                     sigma=gausian_filter_size)
        # make relative
#         s = self.flatField
        s /= s.max()

#         import pylab as plt
#         plt.imshow(self.flatField)
#         plt.colorbar()
#         plt.show()

        return s

    def _fitImg(self, img):
        '''
        fit perspective and size of the input image to the reference image
        '''
        img = imread(img, 'gray')
        if self.bg is not None:
            img = cv2.subtract(img, self.bg)

        (H, _, _, _, _, _, _, n_matches) = self.findHomography(img)
        H_inv = self.invertHomography(H)

        s = self.obj_shape
#         print self._fH,55555555555555
        fit = cv2.warpPerspective(img, H_inv, (s[1], s[0])

                                  #                                    (int( s[1]/self._fH),
                                  # int( s[0]/self._fH) )
                                  )
#         plt.imshow(fit)
#         plt.show()
        return fit, img, H, H_inv, n_matches

    def _findObject(self, img):
        '''
        Create a bounding box around the object within an image
        '''

        from imgProcessor.imgSignal import signalMinimum

        # img is scaled already
        i = img > signalMinimum(img)  # img.max()/2.5
        # filter noise, single-time-effects etc. from mask:
        i = minimum_filter(i, 4)
        return boundingBox(i)


def vignettingFromSameObject(imgs, bg, inPlane_scale_factor=None):
    # TODO: inPlane_scale_factor

    s = ObjectVignettingSeparation(imgs[0], bg)
    for img in imgs[1:]:
        s.addImg(img)
    return s.separate()[:2]


if __name__ == '__main__':
    pass
#     from fancytools.os.PathStr import PathStr
#     from matplotlib import pyplot
    # TODO: generic example
