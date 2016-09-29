from __future__ import division

import numpy as np

def cdf(arr, pos=None):
    '''
    Return the cumulative density function of a given array or
    its intensity at a given position (0-1)
    '''
    
    r = (arr.min(), arr.max())
    hist, bin_edges = np.histogram(arr, bins=2*int(r[1]-r[0]), range=r)
    hist = np.asfarray(hist)/ hist.sum()
    cdf = np.cumsum(hist)
    if pos is None:
        return cdf
    i = np.argmax(cdf>pos)
    return bin_edges[i]