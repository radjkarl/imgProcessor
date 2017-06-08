from __future__ import division

import numpy as np
from numba import jit
from fancytools.math.linspace2 import linspace2


def coarseMaximum(arr, shape):
    '''
    return an array of [shape]
    where every cell equals the localised maximum of the given array [arr]
    at the same (scalled) position
    '''
    ss0, ss1 = shape
    s0, s1 = arr.shape

    pos0 = linspace2(0, s0, ss0, dtype=int)
    pos1 = linspace2(0, s1, ss1, dtype=int)

    k0 = pos0[0]
    k1 = pos1[0]

    out = np.empty(shape, dtype=arr.dtype)
    _calc(arr, out, pos0, pos1, k0, k1, ss0, ss1)
    return out


@jit(nopython=True)
def _calc(arr, out, pos0, pos1, k0, k1, ss0, ss1):
    for i in range(ss0):
        for j in range(ss1):
            p0 = pos0[i]
            p1 = pos1[j]
            val = arr[p0, p1]
            for ii in range(-k0, k0):
                for jj in range(-k1, k1):
                    val2 = arr[p0 + ii, p1 + jj]
                    if val2 > val:
                        val = val2

            out[i, j] = val


if __name__ == '__main__':
    import pylab as plt
    import sys
    s = 300, 400
    s2 = 32, 42
    a = np.fromfunction(lambda x, y: np.sin(x / (0.1 * s[0]))
                        + np.cos(y / (0.01 * s[1])), s)

    b = coarseMaximum(a, s2)

    if 'no_window' not in sys.argv:
        plt.figure('in')
        plt.imshow(a, interpolation='none')
        plt.figure('out')
        plt.imshow(b, interpolation='none')
        plt.show()
