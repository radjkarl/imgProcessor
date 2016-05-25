'''
various image input/output routines
'''
import cv2
import numpy as np

import imgProcessor as ip
from imgProcessor.transformations import transpose, toNoUintArray, toUIntArray


COLOR2CV = {'gray':cv2.IMREAD_GRAYSCALE,
            'all':cv2.IMREAD_COLOR,
            None:cv2.IMREAD_ANYCOLOR
            }


def _changeArrayDType(img, dtype, **kwargs):
    if dtype == 'noUint':
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
    elif isinstance(img, basestring):
        from_file = True
        #grey - 8 bit
        if dtype == None or np.dtype(dtype) != np.uint8:
            c |= cv2.IMREAD_ANYDEPTH
        img2 = cv2.imread(img, c)
        if img2 is None:
            raise IOError('image %s is not existing' %img)
        img = img2
        
    elif color=='gray' and img.ndim == 3:#multi channel img like rgb
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # transform array to uint8 array due to openCV restriction
    if dtype is not None:
        img = _changeArrayDType(img, dtype, cutHigh=False)
    if from_file and ip.ARRAYS_ORDER_IS_XY:
    #if not from_file and not ignore_order and ip.ARRAYS_ORDER_IS_XY: 
        img = cv2.transpose(img)   
    return img


def imwrite(name, arr, **kwargs):
    return cv2.imwrite(name, toUIntArray(arr, **kwargs))


def out(img):
    if ip.ARRAYS_ORDER_IS_XY:
        return transpose(img)
    return img


def out3d(sf):
    '''
    for surface plots
    sf = 3darray[i,j,x,y,z]
    '''
    if ip.ARRAYS_ORDER_IS_XY:
        #transpose values
        sf[:,:,:2] = sf[:,:,1::-1]
        #transpose shape
        return np.transpose(sf, axes=(1,0,2))
    return sf
