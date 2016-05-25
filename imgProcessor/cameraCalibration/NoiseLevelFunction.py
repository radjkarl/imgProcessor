from imgProcessor.imgIO import imread

import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline



def estimateFromImages(imgs1, imgs2=None, mn_mx=None, nbins=100):
    '''
    estimate the noise level function as stDev over image intensity
    from a set of 2 image groups 
    images at the same position have to show
    the identical setup, so
    imgs1[i] - imgs2[i] = noise
    '''
    if imgs2 is None:
        imgs2 = [None]*len(imgs1)
    else:
        assert len(imgs1)==len(imgs2)
    
    y_vals = np.empty((len(imgs1),nbins))
    w_vals = np.zeros((len(imgs1),nbins))
    
    if mn_mx is None:
        print('estimating min and max image value')
        mn = 1e6
        mx = -1e6
        #get min and max image value checking all first images:         
        for n, i1 in enumerate(imgs1):
            print '%s/%s' %(n+1, len(imgs1))
            i1 = imread(i1)
            mmn, mmx = _getMinMax(i1)
            mn = min(mn, mmn)
            mx = mx = max(mx, mmx)
        print('--> min(%s), max(%s)' %(mn,mx))
    else:
        mn,mx = mn_mx
    
    x = None
    print('get noise level function')
    for n,(i1, i2) in enumerate(zip(imgs1,imgs2)):
        print('%s/%s' %(n+1,len(imgs1)))

        i1 = imread(i1)
        if i2 is not None:
            i2 = imread(i2)
    
        x,y,weights, _ = calcNLF(i1, i2, mn_mx_nbins=(mn, mx, nbins), x=x)
        y_vals[n] = y
        w_vals[n] = weights
        
    #filter empty places:
    filledPos = np.sum(w_vals, axis=0)!=0
    w_vals = w_vals[:,filledPos]
    y_vals = y_vals[:,filledPos]
    x = x[filledPos]

    y_avg = np.average(np.nan_to_num(y_vals), 
                       weights=w_vals,
                        axis=0)
    
    w_vals = np.sum(w_vals, axis=0)
    w_vals /= w_vals.sum()

    fitParams, fn, i = _evaluate(x, y_avg, w_vals)
    return x, fn, y_avg, y_vals, w_vals, fitParams,i



def _evaluate(x, y, weights):
    '''
    get the parameters of the, needed by 'function'
    through curve fitting
    '''
    i = _validI(x, y, weights)
    xx = x[i]
    y= y[i]

    try:
        fitParams = _fit(xx, y)
        #bound noise fn to min defined y value:
        minY = function(xx[0], *fitParams)
        fitParams = np.insert(fitParams,0,minY)
        fn = lambda x, minY=minY: boundedFunction(x,*fitParams)
    except RuntimeError:
        print("couldn't fit noise function with filtered indices")
        fitParams = None 
        weights = weights[i]
        fn = smooth(xx,y, weights)
    return  fitParams, fn, i  


def boundedFunction(x, minY, ax, ay):
    '''
    limit [function] to a minimum y value 
    '''
    y = function(x, ax, ay)
    return np.maximum(np.nan_to_num(y),minY)


def function(x, ax, ay):
    '''
    general square root function
    '''
    with np.errstate(invalid='ignore'):
        return ay*(x-ax)**0.5


def _validI(x, y, weights):
    '''
    return indices that have enough data points and are not erroneous 
    '''
    #density filter:
    i = weights > np.median(weights)
    #filter outliers:
    try:
        grad = np.abs(np.gradient(y[i]))
        max_gradient = 4*np.median(grad)
        i[i][grad>max_gradient]=False
    except IndexError:
        pass
    return i
  
    
def _fit(x, y):
    popt, _ = curve_fit(function, x, y, check_finite=False)
    return popt
     

def smooth(x,y,weights):
    '''
    in case the NLF cannot be described by 
    a square root function
    express it as a group of smoothed splines
    '''
    return UnivariateSpline(x, y, w=weights)
    
    
def oneImageNLF(img, img2=None, signal=None): 
    '''
    Estimate the NLF from one or two images of the same kind
    '''
    x, y, weights, signal = calcNLF(img, img2, signal)
    _, fn, _ = _evaluate(x, y, weights)
    return fn, signal
    

def _getMinMax(img):
    '''
    Get the a range of image intensities
    that most pixels are in with
    '''
    av = np.mean(img)
    std = np.std(img)
    #define range for segmentation:
    mn = av-3*std
    mx = av+3*std

    return max(img.min(), mn, 0), min(img.max(),mx)

 
def calcNLF(img, img2=None, signal=None, mn_mx_nbins=None, x=None,
             averageFn='AAD',
             signalFromMultipleImages=False):
    '''
    Calculate the noise level function (NLF) as f(intensity)
    using one or two image.
    The approach for this work is published in JPV##########
    
    img2 - 2nd image taken under same conditions
           used to estimate noise via image difference
    
    signalFromMultipleImages - whether the signal is an average of multiple
        images and not just got from one median filtered image
    '''
    #CONSTANTS:
    #factor Root mead square to average-absolute-difference:
    F_RMS2AAD = (2/np.pi)**-0.5 
    F_NOISE_WITH_MEDIAN = 1+(1.0/3**2)
    N_BINS = 100
    MEDIAN_KERNEL_SIZE = 3

    def _averageAbsoluteDeviation(d):
        return np.mean(np.abs(d))*F_RMS2AAD
    def _rootMeanSquare(d):
        return (d**2).mean()**0.5

    if averageFn == 'AAD':
        averageFn = _averageAbsoluteDeviation
    else:
        averageFn = _rootMeanSquare
    
    img = np.asfarray(img)  
      
    if img2 is None:
        if signal is None:
            signal = median_filter(img, MEDIAN_KERNEL_SIZE)
        if signalFromMultipleImages:
            diff = img - signal
        else:
            #difference between the filtered and original image:     
            diff = (img - signal)*F_NOISE_WITH_MEDIAN
    else:
        img2 = np.asfarray(img2)
        #2**0.5 because noise is subtracted by noise
        #and variance of sum = sum of variance:
        #var(immg1-img2)~2*var(img)
        #std(2*var) = 2**0.5*var**0.5
        diff = (img-img2)/2**0.5
        if signal is None:
            signal = median_filter(0.5*(img+img2), MEDIAN_KERNEL_SIZE)
    if mn_mx_nbins is not None:
        mn, mx, nbins = mn_mx_nbins
        min_len = 0
    else:
        mn, mx = _getMinMax(signal)
        s = img.shape
        min_len = int(s[0]*s[1]*1e-3)
        if min_len < 1:
            min_len = 5
        #number of bins/different intensity ranges to analyse:
        nbins = N_BINS
        if mx - mn < nbins:
            nbins = int(mx - mn)
    #bin width:
    step = (mx-mn)/float(nbins)
    
    #empty arrays:
    y = np.empty(shape=nbins)
    set_x = False
    if x is None:
        set_x = True
        x = np.empty(shape=nbins)
    #give bins with more samples more weight:
    weights = np.zeros(shape=nbins)

    #cur step:
    m = mn
    for n in xrange(nbins):
        #get indices of all pixel with in a bin:
        ind = np.logical_and(signal>=m, signal<=m+step)
        m += step
        d = diff[ind]
        ld = len(d)
        if ld >= min_len:
            weights[n] = ld
            #average absolute deviation (AAD),
            #scaled to RMS:
            y[n] = averageFn(d)
            if set_x:
                x[n] = m - 0.5*step

    return x, y, weights, signal


if __name__ == '__main__':
    import imgProcessor
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    #extract the noise level function from one image:
    f = PathStr(imgProcessor.__file__).dirname().join(
                        'media', 'electroluminescence','EL_module_orig.PNG')
    img = imread(f,'gray')
    fn = oneImageNLF(img)[0]
    x = np.arange(img.min(),img.max())

    plt.plot(x,fn(x))
    plt.show()