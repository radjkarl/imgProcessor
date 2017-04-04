'''
Created on 23 Jan 2017

@author: elkb4
'''
from numba import njit
from numpy import exp

@njit
def numbaGaussian2d(psf, sy, sx):
    '''
    2d Gaussian to be used in numba code
    '''
    ps0, ps1 = psf.shape
    c0,c1 = ps0//2, ps1//2
    ssx = 2*sx**2
    ssy = 2*sy**2
    for i in range(ps0):
        for j in range(ps1):
            psf[i,j]=exp( -( (i-c0)**2/ssy
                            +(j-c1)**2/ssx) )
    psf/=psf.sum()




if __name__ == '__main__':
    import numpy as np
    import sys
    from scipy.ndimage.filters import gaussian_filter
    import pylab as plt


    ksize=4
    size=4*ksize+1

    
    psf = np.empty(shape=(size,size))
    numbaGaussian2d(psf, 2,4)

    if 'no_window' not in sys.argv:
        plt.figure('PSF from this method')
        plt.imshow(psf, interpolation='none', clim=(0,0.03))
        plt.colorbar()
        
        plt.figure('PSF from convolution with Gaussian filter')
        psf2 = np.zeros((size,size))
        psf2[size//2,size//2]=1
        plt.imshow(gaussian_filter(psf2,(2,4)), clim=(0,0.03),interpolation='none')
        plt.colorbar()
        plt.show()