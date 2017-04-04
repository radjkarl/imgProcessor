# coding=utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

import cv2

from scipy.ndimage.filters import minimum_filter, maximum_filter


from fancytools.math.boundingBox import boundingBox
from fancytools.math.MaskedMovingAverage import MaskedMovingAverage

from imgProcessor.imgIO import imread, imwrite
from imgProcessor.utils.decompHomography import decompHomography
from imgProcessor.features.PatternRecognition import PatternRecognition


from imgProcessor.filters.fastFilter import fastFilter
from imgProcessor.utils.getBackground import getBackground
from imgProcessor.camera.LensDistortion import LensDistortion
from fancytools.os.PathStr import PathStr
from imgProcessor.array.subCell2D import subCell2DFnArray


# STARK VEREINFACHEN
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

    def __init__(self, img, bg=None, maxDev=1e-4, maxIter=10, remove_border_size=0,
                 # feature_size=5,
                 cameraMatrix=None, distortionCoeffs=None):  # 20
        """
        Args:
            img (path or array): Reference image
        Kwargs:
            bg (path or array): background image - same for all given images
            maxDev (float): Relative deviation between the last two iteration steps
                            Stop iterative refinement, if deviation is smaller
            maxIter (int): Stop iterative refinement after maxIter steps
        """
        self.lens = None
        if cameraMatrix is not None:
            self.lens = LensDistortion()
            self.lens._coeffs['distortionCoeffs'] = distortionCoeffs
            self.lens._coeffs['cameraMatrix'] = cameraMatrix

        self.maxDev = maxDev
        self.maxIter = maxIter
        self.remove_border_size = remove_border_size
        #self.feature_size = feature_size
        img = imread(img, 'gray')

        self.bg = bg
        if bg is not None:
            self.bg = getBackground(bg)
            if not isinstance(self.bg, np.ndarray):
                self.bg = np.full_like(img, self.bg, dtype=np.uint16)
            else:
                self.bg = self.bg.astype(np.uint16)

            img = cv2.subtract(img, self.bg)

        if self.lens is not None:
            img = self.lens.correct(img, keepSize=True)
        # CREATE TEMPLATE FOR PATTERN COMPARISON:
        pos = self._findObject(img)
        self.obj_shape = img[pos].shape

        PatternRecognition.__init__(self, img[pos])

        self._ff_mma = MaskedMovingAverage(shape=img.shape,
                                           dtype=np.float64)

        self.object = None

        self.Hs = []    # Homography matrices of all fitted images
        self.Hinvs = []  # same, but inverse
        self.fits = []  # all imaged, fitted to reference
        self._fit_masks = []

        self._refined = False

    # TODO: remove that property?
    @property
    def flatField(self):
        return self._ff_mma.avg

    @flatField.setter
    def flatField(self, arr):
        self._ff_mma.avg = arr

    def addImg(self, img, maxShear=0.015, maxRot=100, minMatches=12,
               borderWidth=3):  # borderWidth=100
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
            i = img > self.signal_ranges[-1][0]

            # remove borders (that might have erroneous light):
            i = minimum_filter(i, borderWidth)

            self._ff_mma.update(img, i)

            # create fit img mask:
            mask = fit < self.signal_ranges[-1][0]
            mask = maximum_filter(mask, borderWidth)
            # IGNORE BORDER
            r = self.remove_border_size
            if r:
                mask[:r, :] = 1
                mask[-r:, :] = 1
                mask[:, -r:] = 1
                mask[:, :r] = 1
            self._fit_masks.append(mask)

            # image added
            return fit
        return False

    def error(self, nCells=15):
        '''
        calculate the standard deviation of all fitted images, 
        averaged to a grid
        '''
        s0, s1 = self.fits[0].shape
        aR = s0 / s1
        if aR > 1:
            ss0 = int(nCells)
            ss1 = int(ss0 / aR)
        else:
            ss1 = int(nCells)
            ss0 = int(ss1 * aR)
        L = len(self.fits)

        arr = np.array(self.fits)
        arr[np.array(self._fit_masks)] = np.nan
#         arr = np.ma.array(self.fits, mask=self._fit_masks)
        avg = np.tile(np.nanmean(arr, axis=0), (L, 1, 1))
        arr = (arr - avg) / avg
#         for a in arr:
#             a-=avg
#             a/=avg
#         arr/=avg
#         np.save('aaaaaaabbc3', np.array(avg))
#
#         np.save('aaaaaaabbc1', self.fits)
#         np.save('aaaaaaabbc2', np.array(arr))

        out = np.empty(shape=(L, ss0, ss1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            for n, f in enumerate(arr):
                #                 dev = np.ma.array((f-avg)/avg, mask=mask)
                #                 dev[mask]=np.nan
                out[n] = subCell2DFnArray(f, np.nanmean, (ss0, ss1))

#         np.save('aaaaaaabbc5', out)
#         lk

        return np.nanmean(out**2)**0.5  # np.nanstd(out)#/np.nanmean(out)

#         std = np.nanstd(out, axis=0)
#         return std/np.nanmean(out, axis=0)
#         avg = np.tile(np.nanmean(out, axis=0), (L,1,1))
#         out/=avg
#         return np.nanmean(out**2,axis=0)**0.5

    def separate(self):
        self.flatField = self._createInitialflatField()

        # todo: remove follwing
#         self.init_ff = self.flatField.copy()

        for step in self:
            print('iteration step %s/%s' % (step, self.maxIter))

        # TODO: remove smooth from here - is better be done in post proc.
        smoothed_ff, mask = self.smooth()

        if self.lens is not None:
            smoothed_ff = self.lens.distortImage(smoothed_ff)
            mask = self.lens.distortImage(mask.astype(np.uint8)).astype(bool)

        return smoothed_ff, mask, self.flatField, self.object

    def smooth(self):
        # TODO: there is non nan in the ff img, or?
        mask = self.flatField == 0
        from skimage.filters.rank import median, mean
        from skimage.morphology import disk

        ff = mean(median(self.flatField, disk(5), mask=~mask),
                  disk(13), mask=~mask)

        return ff.astype(float) / ff.max(), mask

#
#     def _mkFitMasks(self):
#         # CREATE IMAGE FIT MASK:
#         for i, f in enumerate(self.fits):
#                 # INDEX FOREGROUND:
#             mask = f < self.signal_ranges[i][0]
#             mask = maximum_filter(mask, self.feature_size)
#
#             # IGNORE BORDER
#             r = self.remove_border_size
#             if r:
#                 mask[:r, :] = 1
#                 mask[-r:, :] = 1
#                 mask[:, -r:] = 1
#                 mask[:, :r] = 1
#             self._fit_masks.append(mask)

    def __iter__(self):
        # use iteration to refine the flatField array

        # keep track of deviation between two iteration steps
        # for break criterion:
        self._last_dev = None
        self.n = 0  # iteration number

#         self._mkFitMasks()

        return self

    def __next__(self):
        # THE IMAGED OBJECT WILL BE AVERAGED FROM ALL
        # INDIVITUAL IMAGES SHOWING THIS OBJECT FROM DIFFERENT POSITIONS:
        obj = MaskedMovingAverage(shape=self.obj_shape)

        with np.errstate(divide='ignore', invalid='ignore'):
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

        new_flatField = s.avg

        # STOP ITERATION?
        # RMSE excluding NaNs:
        dev = np.nanmean((new_flatField[::10, ::10]
                          - self.flatField[::10, ::10])**2)**0.5
        print('residuum: %s' % dev)
        if self.n > self.maxIter or (self._last_dev and (
            (self.n > 4 and dev > self._last_dev)
                or dev < self.maxDev)):
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

        s = fastFilter(self.flatField, f, every)

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

        if self.lens is not None:
            img = self.lens.correct(img, keepSize=True)

        (H, _, _, _, _, _, _, n_matches) = self.findHomography(img)
        H_inv = self.invertHomography(H)

        s = self.obj_shape
        fit = cv2.warpPerspective(img, H_inv, (s[1], s[0])

                                  # (int( s[1]/self._fH),
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


def vignettingFromRandomSteps(imgs, bg, inPlane_scale_factor=None,
                              debugFolder=None, **kwargs):
    '''
    important: first image should shown most iof the device
    because it is used as reference
    '''
    # TODO: inPlane_scale_factor
    if debugFolder:
        debugFolder = PathStr(debugFolder)

    s = ObjectVignettingSeparation(imgs[0], bg,  **kwargs)
    for img in imgs[1:]:
        fit = s.addImg(img)

        if debugFolder and fit is not False:
            imwrite(debugFolder.join('fit_%s.tiff' % len(s.fits)), fit)

    if debugFolder:
        imwrite(debugFolder.join('init.tiff'), s.flatField)

    smoothed_ff, mask, flatField, object = s.separate()

    if debugFolder:
        imwrite(debugFolder.join('object.tiff'), object)
        imwrite(debugFolder.join('flatfield.tiff'), flatField, dtype=float)
        imwrite(debugFolder.join('flatfield_smoothed.tiff'), smoothed_ff,
                dtype=float)

    return smoothed_ff, mask


if __name__ == '__main__':
    pass
#     from fancytools.os.PathStr import PathStr
#     from matplotlib import pyplot
    # TODO: generic example
