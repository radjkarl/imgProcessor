import cv2
import numpy as np
from numba import njit

from scipy.ndimage.filters import maximum_filter, median_filter

from imgProcessor.filters.maskedFilter import maskedFilter
from fancytools.math.linRegressUsingMasked2dArrays \
        import linRegressUsingMasked2dArrays


def temporalSignalStability(imgs, times, down_scale_factor=1):
    '''
    (Electroluminescence) signal is not stable over time
        especially next to cracks.
    This function takes a set of images
    and returns parameters, needed to transform uncertainty 
        to other exposure times using [adjustUncertToExposureTime]
    
    
    return [signal uncertainty] obtained from linear fit to [imgs]
           [average event length] 
           [ascent],[offset] of linear fit
    
    --------
    [imgs] --> corrected EL images captured in sequence
    
    [times] --> absolute measurement times of all [imgs]
                e.g. every image was taken every 60 sec, then 
                times=60,120,180... 
    [down_scale_factor] --> down scale [imgs] to speed up process
    -------

    More information can be found at ...
    ----
    K.Bedrich: Quantitative Electroluminescence Imaging, PhD Thesis, 2017
    Subsection 5.1.4.3: Exposure Time Dependency
    ----
    '''
    imgs = np.asarray(imgs)
    s0, s1, s2 = imgs.shape

    #down scale imgs to speed up process:
    if down_scale_factor > 1:
        s1 //= down_scale_factor
        s2 //= down_scale_factor
        imgs2 = np.empty(shape=(s0, s1, s2))
        for n, c in enumerate(imgs):
            imgs2[n] = cv2.resize(c, (s2, s1), interpolation=cv2.INTER_AREA)
        imgs = imgs2
    
    # linear fit for every point in image set:
    ascent, offset, error = linRegressUsingMasked2dArrays(
                                times, imgs, calcError=True)
    
    # functionally obtained [imgs]:
    fn_imgs = np.array([offset + t * ascent for t in times])
    #difference between [imgs] for fit result:
    diff = imgs - fn_imgs
    diff = median_filter(diff, 5)

    error_t = np.tile(error, (s0, 1, 1))
    # find events: 
    evt = (np.abs(diff) > 0.5 * error_t) 
    # calc average event length:
    avlen = _calcAvgLen(evt, np.empty(shape=evt.shape[1:]))
    
    #cannot calc event length smaller exposure time, so:
    i = avlen == 0
    avlen = maskedFilter(avlen, mask=i, fn='mean', ksize=7, fill_mask=False)
    # remove single px:
    i = maximum_filter(i, 3)
    avlen[i] = 0
    avlen = maximum_filter(avlen, 3)

    i = avlen == 0
    avlen = median_filter(avlen, 3)
    avlen[i] = 0

    return error, avlen, ascent, offset


@njit
def _calcAvgLen(arr, out):
    # calc average length of connected positive elements in z direction
    s0, s1, s2 = arr.shape
    for i in range(s1):
        for j in range(s2):
            last_val = False
            nvals = 0.0
            nclusters = 0.0
            for k in range(s0):
                if arr[k, i, j]:
                    if not last_val:
                        #    last_val = True
                        # else:
                        nclusters += 1
                        last_val = True

                    nvals += 1
                else:
                    last_val = False
            if nclusters == 0:
                out[i, j] = 0
            else:
                out[i, j] = nvals / nclusters
    return out


if __name__ == '__main__':
    import sys
    import pylab as plt
    
    #create synthetic data:    
    res = 100
    device = slice(20,80),slice(20,80)
    signal_t0 = 100
    tignal_t1 = 60
    noise = 5
    nimgs = 15
    #top-left signal fluctuation:
    evt1_pos = slice(30,40),slice(30,40)
    evt1_ampl = 20
    evt1_freq = 10
    #bottom-right signal fluctuation:
    evt2_pos = slice(60,70),slice(60,70)
    evt2_ampl = 30
    evt2_freq = 50  

    imgs =[]    
    s = np.linspace(signal_t0, tignal_t1, nimgs)
    evt1_signal = evt1_ampl*(np.sin(np.linspace(0,evt1_freq, nimgs))+1)/2
    evt2_signal = evt2_ampl*(np.sin(np.linspace(0,evt2_freq, nimgs))+1)/2

    #generate [imgs]
    for si, e1, e2 in zip(s, evt1_signal, evt2_signal):
        img = np.zeros((res,res))
        img[device]=si
        
        img[evt1_pos]+=e1
        img[evt2_pos]+=e2
        
        img+=np.random.normal(scale=noise, size=res**2).reshape(res,res)
        imgs.append(img)

    #####
    error, avlen, ascent, offset = temporalSignalStability(
                                        imgs, np.linspace(0,100,len(imgs)))
    #####
    
    if 'no_window' not in sys.argv:
        f, axes = plt.subplots(1,4)
    
        axes[0].set_title('Signal uncertainty')
        axes[0].imshow(error)
        axes[1].set_title('Event duration')
        axes[1].imshow(avlen)
        axes[2].set_title('Linear fit - ascent')
        axes[2].imshow(ascent)
        axes[3].set_title('Linear fit - offset')
        axes[3].imshow(offset)
        
    
        f, axes = plt.subplots(1,len(imgs))
        f.canvas.set_window_title('Time series [imgs]')
        for i,ax in zip(imgs, axes):
            ax.imshow(i, clim=(0,signal_t0))
            ax.axis('off')
        plt.show()
    
    