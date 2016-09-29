from __future__ import division

import numpy as np


def gaussian(x,  a, b, c, d=0):
    '''
    a -> height of the curve's peak
    b -> position of the center of the peak
    c ->  standard deviation or Gaussian RMS width
    d -> offset
    '''
    return a * np.exp(  -(((x-b)**2 )/ (2*(c**2)))  ) + d 
