from __future__ import division

from numba import njit
import numpy as np


def interpolate2dStructuredCrossAvg(grid, mask, kernel=15, power=2):
    '''
    #######
    usefull if large empty areas need to be filled

    '''

    vals = np.empty(shape=4, dtype=grid.dtype)
    dist = np.empty(shape=4, dtype=np.uint16)
    weights = np.empty(shape=4, dtype=np.float32)
    valid = np.empty(shape=4, dtype=bool)

    return _calc(grid, mask, power, kernel, vals, dist, weights, valid)


@njit
def _localAvg(grid, mask, i, j, gx, gy, kernel):
    val = 0
    n = 0
    xmn = i - kernel
    if xmn < 0:
        xmn = 0
    xmx = i + kernel
    if xmx > gx:
        xmx = gx

    ymn = j - kernel
    if ymn < 0:
        ymn = 0
    ymx = j + kernel
    if ymx > gy:
        ymx = gy
    for xi in range(xmn, xmx + 1):
        for yi in range(ymn, ymx + 1):
            if not mask[xi, yi]:
                val += grid[xi, yi]
                n += 1
    return val / n


@njit
def _calc(grid, mask, power, kernel, vals, dist, weights, valid):
    gx = grid.shape[0]
    gy = grid.shape[1]

    # FOR EVERY PIXEL
    for i in range(gx):
        for j in range(gy):

            if mask[i, j]:
                valid[:] = False

                # look down
                if i > 0:
                    i0 = i
                    while i0:
                        i0 -= 1
                        if not mask[i0, j]:
                            vals[0] = _localAvg(
                                grid, mask, i0, j, gx, gy, kernel)
                            dist[0] = i - i0
                            valid[0] = True

                            break
                # look up
                if i < gx - 1:
                    i0 = i
                    while i0 < gx - 1:
                        i0 += 1
                        if not mask[i0, j]:
                            vals[1] = _localAvg(
                                grid, mask, i0, j, gx, gy, kernel)
                            dist[1] = i0 - i
                            valid[2] = True

                            break
                # look left
                if j > 0:
                    i0 = j
                    while i0:
                        i0 -= 1
                        if not mask[i, i0]:
                            vals[2] = _localAvg(
                                grid, mask, i, i0, gx, gy, kernel)
                            dist[2] = j - i0
                            valid[2] = True

                            break
                # look right
                if i < gy - 1:
                    i0 = j
                    while i0 < gy - 1:
                        i0 += 1
                        if not mask[i, i0]:
                            vals[3] = _localAvg(
                                grid, mask, i, i0, gx, gy, kernel)
                            dist[3] = i0 - j
                            valid[3] = True
                            break
#                 for d in dist:
#                     print(d, 333)
#                 print(44)
                weights[valid] = 1 / dist[valid]**(0.5 * power)
#                 print(dist[valid].sum())
                weights[valid] /= weights[valid].sum()

                grid[i, j] = (vals[valid] * weights[valid]).sum()

    return grid


if __name__ == '__main__':
    import pylab as plt
    import sys
    from time import time

    shape = (200, 200)

    # array with random values:
    arr = np.random.rand(*shape)
    arr += np.tile(np.linspace(5, 10, shape[0]), (shape[1], 1))
    # cut out rect area:
    mask = np.zeros(shape, dtype=bool)
    mask[50:150, 50:150] = 1

    # substituting all cells with mask==True with interpolated value:
    t0 = time()
    arr1 = interpolate2dStructuredCrossAvg(
        arr.copy(), mask, kernel=20, power=2)

    print('time=%f' % (time() - t0))

    if 'no_window' not in sys.argv:

        plt.figure('original')
        arr[mask] = np.nan
        plt.imshow(arr)

        plt.figure('interpolated')

        plt.imshow(arr1)

        plt.show()
