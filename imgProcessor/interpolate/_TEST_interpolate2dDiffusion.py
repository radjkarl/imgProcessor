'''
##############
under development
#####
Created on 4 Apr 2017

@author: elkb4
'''
import numpy as np
from imgProcessor.equations.numbaGaussian2d import numbaGaussian2d
from numba import njit


@njit
def weightedConvolution(img, out, weights, psf):
    a0 = psf.shape[0]
    a1 = psf.shape[1]
    c0, c1 = a0 // 2, a1 // 2
    s0, s1 = img.shape

    # for all px ignoring border for now
    for i in range(c0, s0 - c0):
        for j in range(c1, s1 - c1):

            val = img[i, j]
            for ii in range(a0):
                for jj in range(a1):
                    n_px = img[i - ii + c0, j - jj + c1]
                    w = weights[i - ii + c0, j - jj + c1]
                    val += w * psf[ii, jj] * n_px
            out[i, j] = val


def interpolate2dDiffusion(arr1, arr2, steps=10, diffusivity=0.2):

    psf = np.zeros((5, 5))
    numbaGaussian2d(psf, 1, 1)
#     plt.imshow(psf)
#     plt.show()
    last = arr1

    out = []
    for s in range(steps):
        next = np.zeros_like(arr1)
        diff = diffusivity * (last - arr2)
#         plt.imshow(diff)
#         plt.show()
        weightedConvolution(last, next, diff, psf)

        out.append(next)
        last = next
    return out

if __name__ == '__main__':
    import pylab as plt
    import sys
    from scipy.ndimage.filters import gaussian_filter

    arr1 = np.zeros((100, 100))
    arr1[20:40, 20:40] = 1

    arr2 = np.zeros((100, 100))
    arr2[30:50, 30:50] = 1

    arr1 = gaussian_filter(arr1, 3)
    arr2 = gaussian_filter(arr2, 3)

    if 'no_window' not in sys.argv:
        plt.imshow(arr1 - arr2)
        plt.show()
        out = interpolate2dDiffusion(arr1, arr2)

        for n, o in enumerate(out):
            plt.figure(n)
            plt.imshow(o)
        plt.show()
