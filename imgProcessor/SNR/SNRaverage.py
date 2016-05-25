import numpy as np

from imgProcessor.FitHistogramPeaks import FitHistogramPeaks
from imgProcessor.scaleSignal import getSignalMinimum, hasBackground



def SNRaverage(snr, method='X75', excludeBackground=True, 
               checkBackground=True,
               backgroundLevel=None):
    '''
    average a signal-to-noise map
    :param method:  ['average','X75', 'RMS', 'median']
    - X75: this SNR will be exceeded by 75% of the signal
    :param checkBackground: [bool] - check whether there is actually a 
    background level to exclude
    '''
    if excludeBackground:  
        #get background level
        if backgroundLevel is None: 
            try: 
                f = FitHistogramPeaks(snr).fitParams
                if checkBackground:
                    if not hasBackground(f):
                        excludeBackground = False
                if excludeBackground:
                    backgroundLevel = getSignalMinimum(f)
            except ValueError:
                backgroundLevel = snr.min()
        if excludeBackground:
            snr = snr[snr>=backgroundLevel]
    
    if method == 'RMS':
        avg = (snr**2).mean()**0.5
    
    elif method == 'average':
        avg = snr.mean()
    elif method == 'median':
        avg = np.median(snr)
    
    elif method == 'X75':
        r = (snr.min(), snr.max())
        hist, bin_edges = np.histogram(snr, bins=2*int(r[1]-r[0]), range=r)
        hist = np.asfarray(hist) / hist.sum()
        cdf = np.cumsum(hist)
        i = np.argmax(cdf>0.25)
        avg = bin_edges[i]
    else:
        raise NotImplemented("given SNR average doesn't exist")
    
    return avg
                