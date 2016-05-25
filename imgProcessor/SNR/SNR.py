import numpy as np
from scipy.ndimage.filters import median_filter

from imgProcessor.cameraCalibration.NoiseLevelFunction import oneImageNLF

#factor root-mean-square to average-absolute-deviation (AAD)
#see https://en.wikipedia.org/wiki/Average_absolute_deviation
#citing GEARY1935:
F_RMS2AAD = (2/np.pi)**-0.5 
#factor scaling resulting noise is extracted by the difference
#of an image by its median filtered version
#see BEDRICH 2016 JPV (not jet published):
F_DiffTHroughMedian = (1+(1.0/3**2))


def SNR(img1, img2=None, bg=0,
        noise_level_function=None,
        constant_noise_level=False,
        imgs_to_be_averaged=False):
    '''
    Returns a signal-to-noise-map
    uses algorithm as described in BEDRICH 2016 JPV (not jet published)
    
    @param constant_noise_level = True, to assume noise to be constant
    
    @param imgs_to_be_averaged = True, if SNR is for average(img1, img2)
    
    ''' 
    #dark current subtraction:       
    img1 = np.asfarray(img1) - bg
    
    if img2 is not None:
        img2_exists = True
        img2 = np.asfarray(img2) - bg
        #signal as average on both images
        signal = 0.5*(img1+img2)
    else:
        img2_exists = False
        signal = img1
   
    #denoise:
    signal = median_filter(signal, 3)

    if constant_noise_level:
        if img2_exists:
            #0.5**0.5 because of sum of variances
            noise = 0.5**0.5 *np.mean(np.abs((img1-img2)))*F_RMS2AAD
        else:
            d = (img1-signal)*F_DiffTHroughMedian
            noise = np.mean(np.abs(d))*F_RMS2AAD
    else:
        if noise_level_function is None:
            noise_level_function, _ = oneImageNLF(img1, img2, signal)
        noise = noise_level_function(signal)
        noise[noise<1]=1#otherwise SNR could be higher than image value
    
    if imgs_to_be_averaged:
        #factor of noise reduction if SNR if for average(img1, img2):
        noise *= 0.5**0.5
    
    #background estimation and removal if background not given:
    if bg is 0: 
        bg = getBackgroundLevel(img1)
        signal -= bg
    snr = signal/noise

    #limit to 1, saying at these points signal=noise:
    snr[snr<1]=1
    return snr


def getBackgroundLevel(img):
    #seems to be best one according of no-ref bg comparison
    #as done for SNR article in BEDRICH2016 JPV
    return median_filter(img[10:-10,10:-10],7).min()