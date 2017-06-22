# coding=utf-8
'''
various image transformation functions
'''
from __future__ import division

import numpy as np
import cv2


def toUIntArray(img, dtype=None, cutNegative=True, cutHigh=True,
                range=None, copy=True):
    '''
    transform a float to an unsigned integer array of a fitting dtype
    adds an offset, to get rid of negative values

    range = (min, max) - scale values between given range

    cutNegative - all values <0 will be set to 0
    cutHigh - set to False to rather scale values to fit
    '''
    mn, mx = None, None
    if range is not None:
        mn, mx = range

    if dtype is None:
        if mx is None:
            mx = np.nanmax(img)
        dtype = np.uint16 if mx > 255 else np.uint8

    dtype = np.dtype(dtype)
    if dtype == img.dtype:
        return img

    # get max px value:
    b = {'uint8': 255,
         'uint16': 65535,
         'uint32': 4294967295,
         'uint64': 18446744073709551615}[dtype.name]

    if copy:
        img = img.copy()

    if range is not None:
        img = np.asfarray(img)
        img -= mn
        # img[img<0]=0
        # print np.nanmin(img), np.nanmax(img), mn, mx, range, b

        img *= b / (mx - mn)
        # print np.nanmin(img), np.nanmax(img), mn, mx, range, b
        img = np.clip(img, 0, b)

    else:
        if cutNegative:
            img[img < 0] = 0
        else:
            # add an offset to all values:
            mn = np.min(img)
            if mn < 0:
                img -= mn  # set minimum to 0

        if cutHigh:
            #ind = img > b
            img[img > b] = b
        else:
            # scale values
            mx = np.nanmax(img)
            img = np.asfarray(img) * (float(b) / mx)

    img = img.astype(dtype)

#     if range is not None and cutHigh:
#         img[ind] = b
    return img


def toFloatArray(img):
    '''
    transform an unsigned integer array into a
    float array of the right size
    '''
    _D = {1: np.float32,  # uint8
          2: np.float32,  # uint16
          4: np.float64,  # uint32
          8: np.float64}  # uint64
    return img.astype(_D[img.itemsize])


def toNoUintArray(arr):
    '''
    cast array to the next higher integer array
    if dtype=unsigned integer
    '''
    d = arr.dtype
    if d.kind == 'u':
        arr = arr.astype({1: np.int16,
                          2: np.int32,
                          4: np.int64}[d.itemsize])
    return arr


def isColor(img):
    return img.ndim in (3, 4) and img.shape[-1] in (3, 4)


def toColor(img):
    # color order is assumed to be RGB (red,  green, blue)
    s = img.shape
    if len(s) == 2:  # one gray scale img
        out = np.empty((s[0], s[1], 3))
        out[:, :, 0] = img  # *(1/3.0)#0.114
        out[:, :, 1] = img  # *(1/3.0)#0.587
        out[:, :, 2] = img  # *(1/3.0)#0.299
    elif len(s) == 3:  # mutliple gray scale images
        out = np.empty((s[0], s[1], s[2], 3))
        out[:, :, :, 0] = img  # *(1/3.0)#0.114
        out[:, :, :, 1] = img  # *(1/3.0)#0.587
        out[:, :, :, 2] = img  # *(1/3.0)#0.299
    else:
        # assume is already multilayer color img
        return img
    return out


def toGray(img):
    '''
    weights see
    https://en.wikipedia.org/wiki/Grayscale#Colorimetric_.28luminance-prese
    http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
    '''
    return np.average(img, axis=-1, weights=(0.299,  # red
                                             0.587,  # green
                                             0.114)  # blue
                      ).astype(img.dtype)


def rgChromaticity(img):
    '''
    returns the normalized RGB space (RGB/intensity)
    see https://en.wikipedia.org/wiki/Rg_chromaticity
    '''
    out = _calc(img)
    if img.dtype == np.uint8:
        out = (255 * out).astype(np.uint8)
    return out


def _calc(img):
    out = np.empty_like(img, dtype=float)
    f = img.sum(axis=2)
    for n in range(3):
        out[..., n] = img[..., n] / f
    return out


def monochromaticWavelength(img):
    '''
    TODO##########
    '''
    # peak wave lengths: https://en.wikipedia.org/wiki/RGB_color_model
    out = _calc(img)

    peakWavelengths = (570, 540, 440)  # (r,g,b)
#     s = sum(peakWavelengths)
    for n, p in enumerate(peakWavelengths):
        out[..., n] *= p
    return out.sum(axis=2)


def transpose(img):
    if type(img) in (tuple, list) or img.ndim == 3:
        if img.shape[2] == 3:  # is color
            return np.transpose(img, axes=(1, 0, 2))
        return np.transpose(img, axes=(0, 2, 1))
    else:
        return img.transpose()


def isRot90(shape1, shape2):
    s00, s01 = shape1[:2]
    s10, s11 = shape2[:2]
    return s00 == s11 and s01 == s10


def rot90(img):
    '''
    rotate one or multiple grayscale or color images 90 degrees
    '''
    s = img.shape
    if len(s) == 3:
        if s[2] in (3, 4):  # color image
            out = np.empty((s[1], s[0], s[2]), dtype=img.dtype)
            for i in range(s[2]):
                out[:, :, i] = np.rot90(img[:, :, i])
        else:  # mutliple grayscale
            out = np.empty((s[0], s[2], s[1]), dtype=img.dtype)
            for i in range(s[0]):
                out[i] = np.rot90(img[i])
    elif len(s) == 2:  # one grayscale
        out = np.rot90(img)
    elif len(s) == 4 and s[3] in (3, 4):  # multiple color
        out = np.empty((s[0], s[2], s[1], s[3]), dtype=img.dtype)
        for i in range(s[0]):  # for each img
            for j in range(s[3]):  # for each channel
                out[i, :, :, j] = np.rot90(img[i, :, :, j])
    else:
        NotImplemented
    return out


def applyColorMap(gray, cmap='flame'):
    '''
    like cv2.applyColorMap(im_gray, cv2.COLORMAP_*) but with different color maps
    '''
    # TODO:implement more cmaps
    if cmap != 'flame':
        raise NotImplemented
    # TODO: make better
    mx = 256  # if gray.dtype==np.uint8 else 65535
    lut = np.empty(shape=(256, 3))
    cmap = (
        # taken from pyqtgraph GradientEditorItem
        (0, (0, 0, 0)),
        (0.2, (7, 0, 220)),
        (0.5, (236, 0, 134)),
        (0.8, (246, 246, 0)),
        (1.0, (255, 255, 255))
    )
    # build lookup table:
    lastval, lastcol = cmap[0]
    for step, col in cmap[1:]:
        val = int(step * mx)
        for i in range(3):
            lut[lastval:val, i] = np.linspace(
                lastcol[i], col[i], val - lastval)

        lastcol = col
        lastval = val

    s0, s1 = gray.shape
    out = np.empty(shape=(s0, s1, 3), dtype=np.uint8)

    for i in range(3):
        out[..., i] = cv2.LUT(gray, lut[:, i])
    return out
