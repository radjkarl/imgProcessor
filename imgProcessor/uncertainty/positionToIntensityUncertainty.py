import numpy as np
from numpy import isnan
from numba import jit
from imgProcessor.equations.numbaGaussian2d import numbaGaussian2d


@jit(nopython=True)
def _calc_constPSF(image, sint, sx, sy, psf, ksize):
    ax = psf.shape[0]
    ay = psf.shape[1]
    cx, cy = ax // 2, ax // 2

    numbaGaussian2d(psf, sx, sy)
    s0, s1 = image.shape[:2]
    # for all px ignoring border for now
    for i in range(ksize, s0 - ksize):
        for j in range(ksize, s1 - ksize):
            c_px = image[i, j]
            if not isnan(c_px):
                sdev = 0.0
                for ii in range(ax):
                    for jj in range(ay):
                        n_px = image[i - ii + cx, j - jj + cy]
                        sdev += psf[ii, jj] * (n_px - c_px)**2

                sint[i, j] = (sdev)**0.5


@jit(nopython=True)
def _calc_variPSF(image, sint, sx, sy, psf, ksize):
    ax = psf.shape[0]
    ay = psf.shape[1]
    cx, cy = ax // 2, ax // 2

    # for all px ignoring border for now
    for i in range(ksize, image.shape[0] - ksize):
        for j in range(ksize, image.shape[1] - ksize):
            c_px = image[i, j]
            if not isnan(c_px):

                numbaGaussian2d(psf, sx[i, j], sy[i, j])

                sdev = 0.0
                for ii in range(ax):
                    for jj in range(ay):

                        n_px = image[i - ii + cx, j - jj + cy]
                        sdev += psf[ii, jj] * (n_px - c_px)**2
                sint[i, j] = (sdev)**0.5


def positionToIntensityUncertainty(image, sx, sy, kernelSize=None):
    '''
    calculates the estimated standard deviation map from the changes
    of neighbouring pixels from a center pixel within a point spread function
    defined by a std.dev. in x and y taken from the (sx, sy) maps 

    sx,sy -> either 2d array of same shape as [image]
             of single values
    '''
    psf_is_const = not isinstance(sx, np.ndarray)
    if not psf_is_const:
        assert image.shape == sx.shape == sy.shape, \
            "Image and position uncertainty maps need to have same size"
        if kernelSize is None:
            kernelSize = _kSizeFromStd(max(sx.max(), sy.max()))
    else:
        assert type(sx) in (int, float) and type(sx) in (int, float), \
            "Image and position uncertainty values need to be int OR float"
        if kernelSize is None:
            kernelSize = _kSizeFromStd(max(sx, sy))

    if image.dtype.kind == 'u':
        image = image.astype(int)  # otherwise stack overflow through uint
    size = kernelSize // 2
    if size < 1:
        size = 1
    kernelSize = 1 + 2 * size
    # array to be filled by individual psf of every pixel:
    psf = np.zeros((kernelSize, kernelSize))
    # intensity uncertainty as stdev:
    sint = np.zeros(image.shape)
    if psf_is_const:
        _calc_constPSF(image, sint, sx, sy, psf, size)
    else:
        _calc_variPSF(image, sint, sx, sy, psf, size)
    return sint


def _kSizeFromStd(std):
    return max(3, 4 * std + 1)


def _coarsenImage(image, f):
    '''
    seems to be a more precise (but slower)
    way to down-scale an image
    '''
    from skimage.morphology import square
    from skimage.filters import rank
    from skimage.transform._warps import rescale
    selem = square(f)
    arri = rank.mean(image, selem=selem)
    return rescale(arri, 1 / f, order=0)


def positionToIntensityUncertaintyForPxGroup(image, std, y0, y1, x0, x1):
    '''
    like positionToIntensityUncertainty
    but calculated average uncertainty for an area [y0:y1,x0:x1]
    '''
    fy, fx = y1 - y0, x1 - x0
    if fy != fx:
        raise Exception('averaged area need to be square ATM')
    image = _coarsenImage(image, fx)
    k = _kSizeFromStd(std)
    y0 = int(round(y0 / fy))
    x0 = int(round(x0 / fx))
    arr = image[y0 - k:y0 + k, x0 - k:x0 + k]
    U = positionToIntensityUncertainty(arr, std / fx, std / fx)
    return U[k:-k, k:-k]


if __name__ == '__main__':
    import sys
    import imgProcessor
    from imgProcessor.imgIO import imread
    from fancytools.os.PathStr import PathStr
    from pylab import plt
    d = PathStr(imgProcessor.__file__).dirname().join(
        'media', 'electroluminescence')

    # STITCH BOTTOM
    img = imread(d.join('EL_cell_cracked.png'), 'gray')
    img = img[::2, ::2]  # speed up for this test case

    # CASE 1: constant position uncertainty:
    stdx = 9
    stdy = 3

    sint = positionToIntensityUncertainty(img, stdx, stdy)

    # CASE2: variable position uncertainty:
    # x,y 0...15
    stdx2 = np.fromfunction(lambda x, y: x * y, img.shape)
    stdx2 /= stdx2[-1, -1] / 9
    stdy2 = np.fromfunction(lambda x, y: x * y, img.shape)
    stdy2 /= stdy2[-1, -1] / 9
    stdy2 = stdy2[::-1, ::-1]  # flip content twice

    sint2 = positionToIntensityUncertainty(img, stdx2, stdy2, 21)

    if 'no_window' not in sys.argv:
        plt.figure('input')
        plt.imshow(img)
        plt.colorbar()

        plt.figure('output for const. position uncertainty (x%s,y%s)' %
                   (stdx, stdy))
        plt.imshow(sint)
        plt.colorbar()

        plt.figure('output for var. position uncertainty 0...15')
        plt.imshow(sint2)
        plt.colorbar()

        plt.show()
