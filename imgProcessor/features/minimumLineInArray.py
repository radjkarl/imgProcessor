from __future__ import division
from __future__ import print_function

import numpy as np
from numba import jit
from scipy.ndimage import center_of_mass


@jit(nopython=True)
def _lineSumXY(x, res, sub, f):
    s0 = sub.shape[0]
    sx = x.shape[0]
    hs = (s0 - 1) * 0.5

    for i in range(s0):
        ff = 1 - f * (abs(i - hs) / hs)
        for j in range(s0):
            c = float(i)
            d = float(j - i) / sx
            val = 0.0
            for n in range(sx):
                val += sub[int(round(c)), x[n]]
                c += d
            res[i, j] = val * ff


def minimumLineInArray(arr, relative=False, f=0,
                       refinePosition=True,
                       max_pos=100,
                       return_pos_arr=False,
                       # order=2
                       ):
    '''
    find closest minimum position next to middle line
    relative: return position relative to middle line
    f: relative decrease (0...1) - setting this value close to one will
       discriminate positions further away from the center
    ##order: 2 for cubic refinement
    '''
    s0, s1 = arr.shape[:2]
    if max_pos >= s1:
        x = np.arange(s1)
    else:
        # take fewer positions within 0->(s1-1)
        x = np.rint(np.linspace(0, s1 - 1, min(max_pos, s1))).astype(int)
    res = np.empty((s0, s0), dtype=float)

    _lineSumXY(x, res, arr, f)

    if return_pos_arr:
        return res

    # best integer index
    i, j = np.unravel_index(np.nanargmin(res), res.shape)

    if refinePosition:
        try:
            sub = res[i - 1:i + 2, j - 1:j + 2]
            ii, jj = center_of_mass(sub)
            if not np.isnan(ii):
                i += (ii - 1)
            if not np.isnan(jj):
                j += (jj - 1)
        except TypeError:
            pass


    if not relative:
        return i, j

    hs = (s0 - 1) / 2
    return i - hs, j - hs



#     from imgProcessor.imgIO import imread
# #     img = imread(r'C:\Users\elkb4\Desktop\PhD\Thesis\materials\EL_mod_dist_gradH.png')
#     img = np.load('inp.npy').T
#     arr = img[35:645,327:650].T
#     
#     pos = minimumLineInArray(arr, return_pos_arr=True)
#     i,j = minimumLineInArray(arr)
#     pos2 = minimumLineInArray(-arr, return_pos_arr=True)
#     i2,j2 = minimumLineInArray(-arr)
#     
#     print (i,j,i2,j2)
#     np.save('aa', arr)
#     np.save('bb', pos)
#     np.save('cc', pos2)


if __name__ == '__main__':
    import pylab as plt
    import sys
    

 
#     np.random.seed(100)
    s0, s1 = 31, 200

    mid = (s0 - 1) / 2
    errors0 = []
    errors1 = []

    x = np.arange(s1)

    for var in range(1, 10):
        arr = np.random.rand(s0, s1)

        # draw a minimum line:
        y = np.linspace(mid - var, mid + var, s1, dtype=int)
        arr[y, x] -= 2

        # find line:
        d0, d1 = minimumLineInArray(arr, relative=True)
        e0 = abs(var + d0)
        e1 = abs(var - d1)
        errors0.append(e0)
        errors1.append(e1)

        assert e0 < 1 and e1 < 1, 'error too high: %s, %s' % (e0, e1)

    print('error0[px] mean:%s, std:%s' % (np.mean(errors0), np.std(errors0)))
    print('error1[px] mean:%s, std:%s' % (np.mean(errors1), np.std(errors1)))

    if 'no_window' not in sys.argv:
        # show last result
        y0, y1 = minimumLineInArray(arr)
        yfit = (y0, y1)
        xfit = (0, s1)
        # plot
        plt.figure('given array')
        plt.imshow(arr)
        plt.axes().set_aspect('auto')

        plt.figure('found line')
        plt.imshow(arr)
        plt.plot(xfit, yfit, linewidth=10, linestyle='--')
        plt.axes().set_aspect('auto')

        plt.show()
