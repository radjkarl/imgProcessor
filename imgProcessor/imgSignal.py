from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from imgProcessor.imgIO import imread
from imgProcessor.measure.FitHistogramPeaks import FitHistogramPeaks
from fancytools.math.findXAt import findXAt
# from scipy.optimize.minpack import curve_fit

MAX_SIZE = 700


def scaleSignalCut(img, ratio, nbins=100):
    '''
    scaling img cutting x percent of top and bottom part of histogram
    '''
    start, stop = scaleSignalCutParams(img, ratio, nbins)
    img = img - start
    img /= (stop - start)
    return img


def _toSize(img):
    fac = MAX_SIZE / max(img.shape)
    if fac < 1:
        try:
            return cv2.resize(img, (0, 0), fx=fac, fy=fac,
                              interpolation=cv2.INTER_AREA)
        except cv2.error:
            # cv2.error: ..\..\..\modules\imgproc\src\imgwarp.cpp:3235: error:
            # (-215) dsize.area() > 0 in function cv::resize
            return cv2.resize(img.T, (0, 0), fx=fac, fy=fac,
                              interpolation=cv2.INTER_AREA).T
    return img


def scaleSignalCutParams(img, ratio=0.01, nbins=100):
    img = _toSize(img)
    try:
        h, bins = np.histogram(img, nbins)
    except ValueError:  # img contains NaN
        h, bins = np.histogram(img[np.isfinite(img)], nbins)

    h = np.cumsum(h).astype(float)
    h -= h.min()
    h /= h[-1]

    b0 = bins[0]
    bins = bins[1:]
    bins += 0.5 * (bins[0] - b0)

    try:
        start = findXAt(bins, h, ratio)
    except IndexError:
        start = bins[0]
    try:
        stop = findXAt(bins, h, 1 - ratio)
    except IndexError:
        stop = bins[-1]

    return start, stop


def scaleSignal(img, fitParams=None,
                backgroundToZero=False, reference=None):
    '''
    scale the image between...

    backgroundToZero=True -> 0 (average background) and 1 (maximum signal)
    backgroundToZero=False -> signal+-3std

    reference -> reference image -- scale image to fit this one

    returns:
    scaled image
    '''
    img = imread(img)
    if reference is not None:
        #         def fn(ii, m,n):
        #             return ii*m+n
        #         curve_fit(fn, img[::10,::10], ref[::10,::10])

        low, high = signalRange(img, fitParams)
        low2, high2 = signalRange(reference)
        img = np.asfarray(img)
        ampl = (high2 - low2) / (high - low)
        img -= low
        img *= ampl
        img += low2
        return img
    else:
        offs, div = scaleParams(img, fitParams, backgroundToZero)
        img = np.asfarray(img) - offs
        img /= div
        print('offset: %s, divident: %s' % (offs, div))
        return img


def getBackgroundRange(fitParams):
    '''
    return minimum, average, maximum of the background peak
    '''
    smn, _, _ = getSignalParameters(fitParams)

    bg = fitParams[0]
    _, avg, std = bg
    bgmn = max(0, avg - 3 * std)

    if avg + 4 * std < smn:
        bgmx = avg + 4 * std
    if avg + 3 * std < smn:
        bgmx = avg + 3 * std
    if avg + 2 * std < smn:
        bgmx = avg + 2 * std
    else:
        bgmx = avg + std
    return bgmn, avg, bgmx


def hasBackground(fitParams):
    '''
    compare the height of putative bg and signal peak
    if ratio if too height assume there is no background
    '''
    signal = getSignalPeak(fitParams)
    bg = getBackgroundPeak(fitParams)
    if signal == bg:
        return False
    r = signal[0] / bg[0]
    if r < 1:
        r = 1 / r
    return r < 100


def signalMinimum2(img, bins=None):
    '''
    minimum position between signal and background peak
    '''
    f = FitHistogramPeaks(img, bins=bins)
    i = signalPeakIndex(f.fitParams)
    spos = f.fitParams[i][1]
#     spos = getSignalPeak(f.fitParams)[1]
#     bpos = getBackgroundPeak(f.fitParams)[1]
    bpos = f.fitParams[i - 1][1]
    ind = np.logical_and(f.xvals > bpos, f.xvals < spos)
    try:
        i = np.argmin(f.yvals[ind])
        return f.xvals[ind][i]
    except ValueError as e:
        if bins is None:
            return signalMinimum2(img, bins=400)
        else:
            raise e


def signalMinimum(img, fitParams=None, n_std=3):
    '''
    intersection between signal and background peak
    '''
    if fitParams is None:
        fitParams = FitHistogramPeaks(img).fitParams
    assert len(fitParams) > 1, 'need 2 peaks so get minimum signal'

    i = signalPeakIndex(fitParams)
    signal = fitParams[i]
    bg = getBackgroundPeak(fitParams)
    smn = signal[1] - n_std * signal[2]
    bmx = bg[1] + n_std * bg[2]
    if smn > bmx:
        return smn
    # peaks are overlapping
    # define signal min. as intersection between both Gaussians

    def solve(p1, p2):
        s1, m1, std1 = p1
        s2, m2, std2 = p2
        a = (1 / (2 * std1**2)) - (1 / (2 * std2**2))
        b = (m2 / (std2**2)) - (m1 / (std1**2))
        c = (m1**2 / (2 * std1**2)) - (m2**2 / (2 * std2**2)) - \
            np.log(((std2 * s1) / (std1 * s2)))
        return np.roots([a, b, c])
    i = solve(bg, signal)
    try:
        return i[np.logical_and(i > bg[1], i < signal[1])][0]
    except IndexError:
        # this error shouldn't occur... well
        return max(smn, bmx)


def getSignalMinimum(fitParams, n_std=3):
    assert len(fitParams) > 0, 'need min. 1 peak so get minimum signal'
    if len(fitParams) == 1:
        signal = fitParams[0]
        return signal[1] - n_std * signal[2]

    i = signalPeakIndex(fitParams)
    signal = fitParams[i]
    bg = fitParams[i - 1]
    #bg = getBackgroundPeak(fitParams)
    smn = signal[1] - n_std * signal[2]
    bmx = bg[1] + n_std * bg[2]
    if smn > bmx:
        return smn
    # peaks are overlapping
    # define signal min. as intersection between both Gaussians

    def solve(p1, p2):
        s1, m1, std1 = p1
        s2, m2, std2 = p2
        a = (1 / (2 * std1**2)) - (1 / (2 * std2**2))
        b = (m2 / (std2**2)) - (m1 / (std1**2))
        c = (m1**2 / (2 * std1**2)) - (m2**2 / (2 * std2**2)) - \
            np.log(((std2 * s1) / (std1 * s2)))
        return np.roots([a, b, c])

    i = solve(bg, signal)
    try:
        return i[np.logical_and(i > bg[1], i < signal[1])][0]
    except IndexError:
        # something didnt work out - fallback
        return smn


def getSignalParameters(fitParams, n_std=3):
    '''
    return minimum, average, maximum of the signal peak
    '''
    signal = getSignalPeak(fitParams)
    mx = signal[1] + n_std * signal[2]
    mn = signal[1] - n_std * signal[2]
    if mn < fitParams[0][1]:
        mn = fitParams[0][1]  # set to bg
    return mn, signal[1], mx


def signalStd(img):
    fitParams = FitHistogramPeaks(img).fitParams
    signal = getSignalPeak(fitParams)
    return signal[2]


def backgroundMean(img, fitParams=None):
    try:
        if fitParams is None:
            fitParams = FitHistogramPeaks(img).fitParams
        bg = getBackgroundPeak(fitParams)
        return bg[1]

    except Exception as e:
        print(e)
        # in case peaks were not found:
        return img.mean()


def signalRange(img, fitParams=None, nSigma=3):
    try:
        if fitParams is None:
            fitParams = FitHistogramPeaks(img).fitParams
        signPeak = getSignalPeak(fitParams)
        return (signalMinimum(img, fitParams, nSigma),
                signPeak[1] + nSigma * signPeak[2])
# return (signPeak[1] - nSigma*signPeak[2],signPeak[1] +
# nSigma*signPeak[2])

    except Exception as e:
        print(e)
        # in case peaks were not found:
        s = img.std()
        m = img.mean()
        return m - nSigma * s, m + nSigma * s


def scaleParamsFromReference(img, reference):
    # saving startup time:
    from scipy.optimize import curve_fit

    def ff(arr):
        arr = imread(arr, 'gray')
        if arr.size > 300000:
            arr = arr[::10, ::10]
        m = np.nanmean(arr)
        s = np.nanstd(arr)
        r = m - 3 * s, m + 3 * s
        b = (r[1] - r[0]) / 5
        return arr, r, b

    img, imgr, imgb = ff(img)
    reference, refr, refb = ff(reference)

    nbins = np.clip(15, max(imgb, refb), 50)

    refh = np.histogram(reference, bins=nbins, range=refr)[
        0].astype(np.float32)
    imgh = np.histogram(img, bins=nbins, range=imgr)[0].astype(np.float32)

    import pylab as plt
    plt.figure(1)
    plt.plot(refh)

    plt.figure(2)
    plt.plot(imgh)
    plt.show()

    def fn(x, offs, div):
        return (x - offs) / div

    params, fitCovariances = curve_fit(fn, refh, imgh, p0=(0, 1))
    perr = np.sqrt(np.diag(fitCovariances))
    print('error scaling to reference image: %s' % perr[0])
    # if perr[0] < 0.1:
    return params[0], params[1]


def scaleParams(img, fitParams=None):
    low, high = signalRange(img, fitParams)
    offs = low
    div = high - low

    return offs, div


def getBackgroundPeak(fitParams):
    return fitParams[0]


def getSignalPeak(fitParams):
    i = signalPeakIndex(fitParams)
    return fitParams[i]


def signalPeakIndex(fitParams):
    if len(fitParams) == 1:
        i = 0
    else:

        # find categorical signal peak as max(peak height*standard deviation):
        sizes = [pi[0] * pi[2] for pi in fitParams[1:]]
        # signal peak has to have positive avg:
        for n, p in enumerate(fitParams[1:]):
            if p[1] < 0:
                sizes[n] = 0
        i = np.argmax(sizes) + 1
    return i


if __name__ == '__main__':
    import sys
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor

    img = imread(PathStr(imgProcessor.__file__).dirname().join(
        'media', 'electroluminescence', 'EL_module_orig.PNG'), 'gray')

    print('EL signal within range of %s' % str(signalRange(img)))
    print('EL signal minimum = %s' % signalMinimum(img))

    if 'no_window' not in sys.argv:
        plt.imshow(img)
        plt.colorbar()
        plt.show()
