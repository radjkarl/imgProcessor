
import numpy as np
from numba import njit



def nan_maximum_filter(arr, ksize):
    '''
    same as scipy.filters.maximum_filter
    but working excluding nans
    '''
    out = np.empty_like(arr)
    _calc(arr, out, ksize//2)
    return out


@njit
def _calc(arr, out, ksize):
    gx = arr.shape[0]
    gy = arr.shape[1]
    for i in range(gx):
        for j in range(gy):   
           
            xmn = i-ksize
            if xmn < 0:
                xmn = 0
            xmx = i+ksize
            if xmx > gx:
                xmx = gx
                
            ymn = j-ksize
            if ymn < 0:
                ymn = 0
            ymx = j+ksize
            if ymx > gy:
                ymx = gy
                
            out[i,j] = np.nanmax(arr[xmn:xmx,ymn:ymx])
                
    