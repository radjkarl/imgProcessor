'''
various image input/output routines
'''
import cv2
import numpy as np
from six import string_types

from imgProcessor.transformations import toNoUintArray, toUIntArray, toGray
from PIL import Image  # todo: save 32float TIFF without needing PIL


# TODO:
# from imgProcessor import reader
# READERS = {'elbin':reader.elbin}


def _changeArrayDType(img, dtype, **kwargs):
    if dtype == 'noUint':
        return toNoUintArray(img)
    if issubclass(np.dtype(dtype).type, np.integer):
        return toUIntArray(img, dtype, **kwargs)
    return img.astype(dtype)


# def bitDepth(path, img=None):
#     '''
#     there are no python filetypes between 8bit and 16 bit
#     so, to find out whether an image is 12 or 14 bit resolved
#     we need to check actual file size and image shape
#     '''
#     if img is None:
#         img = imread(img)
#     size = os.path.getsize(path)*8
#     print (size, img.size,8888888,img.shape,  size/img.size)
#     kh
#     return size/img.size


def imread(img, color=None, dtype=None):
    '''
    dtype = 'noUint', uint8, float, 'float', ...
    '''
    COLOR2CV = {'gray': cv2.IMREAD_GRAYSCALE,
                'all': cv2.IMREAD_COLOR,
                None: cv2.IMREAD_ANYCOLOR
                }
    c = COLOR2CV[color]
    if callable(img):
        img = img()
    elif isinstance(img, string_types):
        #         from_file = True
        #         try:
        #             ftype = img[img.find('.'):]
        #             img = READERS[ftype](img)[0]
        #         except KeyError:
        # open with openCV
        # grey - 8 bit
        if dtype in (None, "noUint") or np.dtype(dtype) != np.uint8:
            c |= cv2.IMREAD_ANYDEPTH
        img2 = cv2.imread(img, c)
        if img2 is None:
            raise IOError("image '%s' is not existing" % img)
        img = img2

    elif color == 'gray' and img.ndim == 3:  # multi channel img like rgb
        # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #cannot handle float64
        img = toGray(img)
    # transform array to uint8 array due to openCV restriction
    if dtype is not None:
        if isinstance(img, np.ndarray):
            img = _changeArrayDType(img, dtype, cutHigh=False)

    return img


def imwrite(name, arr, dtype=None, **kwargs):
    if dtype in (float, np.float64, np.float32):
        # save as 32bit float tiff
        Image.fromarray(np.asfarray(arr)).save(name)
    else:
        return cv2.imwrite(name, toUIntArray(arr, dtype=dtype, **kwargs))
