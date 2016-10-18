'''
various image input/output routines
'''
import numpy as np
import cv2
from six import string_types
from imgProcessor import ARRAYS_ORDER_IS_XY

from imgProcessor.transformations import transpose, toNoUintArray, toUIntArray
from PIL import Image

# from imgProcessor import reader

COLOR2CV = {'gray':cv2.IMREAD_GRAYSCALE,
            'all':cv2.IMREAD_COLOR,
            None:cv2.IMREAD_ANYCOLOR
            }

# READERS = {'elbin':reader.elbin}


def _changeArrayDType(img, dtype, **kwargs):
    if repr(dtype) == 'noUint':
        return toNoUintArray(img)
    if issubclass(np.dtype(dtype).type, np.integer):
        return toUIntArray(img, dtype, **kwargs)
    return img.astype(dtype)


def imread(img, color=None, dtype=None, ignore_order=False):
    '''
    dtype = 'noUint', uint8, float, 'float', ...
    '''
    c = COLOR2CV[color]
    from_file = False
    if callable(img):
        img = img()
    elif isinstance(img, string_types):
        from_file = True
#         try:        
#             ftype = img[img.find('.'):]
#             img = READERS[ftype](img)[0]
#         except KeyError:
        #open with openCV
        #grey - 8 bit
        if dtype == None or np.dtype(dtype) != np.uint8:
            c |= cv2.IMREAD_ANYDEPTH
        img2 = cv2.imread(img, c)
        if img2 is None:
            raise IOError("image '%s' is not existing" %img)
        img = img2
        
    elif color=='gray' and img.ndim == 3:#multi channel img like rgb
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # transform array to uint8 array due to openCV restriction
    if dtype is not None:
        if isinstance(img, np.ndarray):
            img = _changeArrayDType(img, dtype, cutHigh=False)    
    if from_file and ARRAYS_ORDER_IS_XY:
    #if not from_file and not ignore_order and ip.ARRAYS_ORDER_IS_XY: 
        img = cv2.transpose(img)   
    return img


def imwrite(name, arr, dtype=None, **kwargs):
    if dtype in (float, np.float64, np.float32):
        # save as 32bit float tiff
        Image.fromarray(arr).save(name)
    else:
        return cv2.imwrite(name, toUIntArray(arr, **kwargs))


def out(img):
    if ARRAYS_ORDER_IS_XY:
        return transpose(img)
    return img


def out3d(sf):
    '''
    for surface plots
    sf = 3darray[i,j,x,y,z]
    '''
    if ARRAYS_ORDER_IS_XY:
        #transpose values
        sf[:,:,:2] = sf[:,:,1::-1]
        #transpose shape
        return np.transpose(sf, axes=(1,0,2))
    return sf
