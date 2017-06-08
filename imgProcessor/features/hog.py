from __future__ import division

import cv2
import numpy as np

from scipy.ndimage.filters import convolve
from imgProcessor.array.subCell2D import subCell2DSlices,\
    subCell2DFnArray


def _angles(orientations):
    # return angles for given number of orientations
    # EXAMPLE
        # IN: 4
        # OUt: [ 0., 0.78,1.57,2.35]
    return np.arange(orientations) * (np.pi / orientations)


def _mkConvKernel(ksize, orientations):
    # create line shaped kernels, like [ | / - \ ] for 4 orientations
    assert ksize[0] % 2 and ksize[1] % 2

    k0, k1 = ksize
    mx, my = (k0 // 2) + 1, (k1 // 2) + 1
    kernel = np.empty((orientations, k0, k1))
    for i, a in enumerate(_angles(orientations)):
        # make line kernel
        x = int(round(4 * np.cos(a) * k0))
        y = int(round(4 * np.sin(a) * k1))
        k = np.zeros((2 * k0, 2 * k1), dtype=np.uint8)
        cv2.line(k, (-x + k0, -y + k1), (x + k0, y + k1),
                 255,
                 thickness=1, lineType=cv2.LINE_AA)
        # resize and scale 0-1:
        ki = k[mx:mx + k0, my:my + k1].astype(float) / 255
        kernel[i] = ki / ki.sum()
    return kernel


def hog(image, orientations=8, ksize=(5, 5)):
    '''
    returns the Histogram of Oriented Gradients

    :param ksize: convolution kernel size as (y,x) - needs to be odd
    :param orientations: number of orientations in between rad=0 and rad=pi

    similar to http://scikit-image.org/docs/dev/auto_examples/plot_hog.html
    but faster and with less options
    '''
    s0, s1 = image.shape[:2]

    # speed up the process through saving generated kernels:
    try:
        k = hog.kernels[str(ksize) + str(orientations)]
    except KeyError:
        k = _mkConvKernel(ksize, orientations)
        hog.kernels[str(ksize) + str(orientations)] = k

    out = np.empty(shape=(s0, s1, orientations))
    image[np.isnan(image)] = 0

    for i in range(orientations):
        out[:, :, i] = convolve(image, k[i])
    return out
hog.kernels = {}


def visualize(hog, grid=(10, 10), radCircle=None):
    '''
    visualize HOG as polynomial around cell center
        for [grid] * cells
    '''
    s0, s1, nang = hog.shape
    angles = np.linspace(0, np.pi, nang + 1)[:-1]
    # center of each sub array:
    cx, cy = s0 // (2 * grid[0]), s1 // (2 * grid[1])
    # max. radius of polynomial around cenetr:
    rx, ry = cx, cy
    # for drawing a position indicator (circle):
    if radCircle is None:
        radCircle = max(1, rx // 10)
    # output array:
    out = np.zeros((s0, s1), dtype=np.uint8)
    # point of polynomial:
    pts = np.empty(shape=(1, 2 * nang, 2), dtype=np.int32)
    # takes grid[0]*grid[1] sample HOG values:
    samplesHOG = subCell2DFnArray(hog, lambda arr: arr[cx, cy], grid)
    mxHOG = samplesHOG.max()
    # sub array slices:
    slices = list(subCell2DSlices(out, grid))
    m = 0
    for m, hhh in enumerate(samplesHOG.reshape(grid[0] * grid[1], nang)):
        hhmax = hhh.max()
        hh = hhh / hhmax
        sout = out[slices[m][2:4]]
        for n, (o, a) in enumerate(zip(hh, angles)):
            pts[0, n, 0] = cx + np.cos(a) * o * rx
            pts[0, n, 1] = cy + np.sin(a) * o * ry
            pts[0, n + nang, 0] = cx + np.cos(a + np.pi) * o * rx
            pts[0, n + nang, 1] = cy + np.sin(a + np.pi) * o * ry

        cv2.fillPoly(sout, pts, int(255 * hhmax / mxHOG))
        cv2.circle(sout, (cx, cy), radCircle, 0, thickness=-1)

    return out


if __name__ == '__main__':
    import sys
    from imgProcessor.imgIO import imread
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    from imgProcessor.imgSignal import scaleSignalCut

    p = PathStr(imgProcessor.__file__).dirname().join(
        'media', 'electroluminescence')
    img = imread(p.join('EL_cell_cracked.PNG'), 'gray')
    k = (45, 45)
    o = 4

    ##
    dimg = np.abs(cv2.Laplacian(img, cv2.CV_64F))
    h = hog(dimg, ksize=k, orientations=o)
    m = visualize(h, (30, 40))
    ##

    if 'no_window' not in sys.argv:
        dimg = scaleSignalCut(dimg, 0.02)
        plt.figure('input')
        plt.imshow(img)

        plt.figure('gradient image with HOG grid')
        plt.imshow(dimg, clim=(0, 1), cmap='gray')
        m = np.asfarray(m)
        m[m == 0] = np.nan
        plt.imshow(m,  cmap='viridis',
                   alpha=.8,
                   interpolation='none')

        plt.figure('maximum HOG')
        plt.imshow(h.max(axis=2))

        plt.show()
