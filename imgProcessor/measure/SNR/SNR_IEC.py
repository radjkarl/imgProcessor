import numpy as np

def SNR_IEC(i1,i2,ibg=0, allow_color_images=False):
    '''
    Calculate the averaged signal-to-noise ratio SNR50
    as defined by IEC NP 60904-13
    
    needs 2 reference EL images and one background image    
    '''
    #ensure images are type float64 (double precision):
    i1 = np.asfarray(i1)
    i2 = np.asfarray(i2)
    if ibg is not 0:
        ibg = np.asfarray(ibg)
        assert i1.shape == ibg.shape, 'all input images need to have the same resolution'
    
    assert i1.shape == i2.shape, 'all input images need to have the same resolution'
    if not allow_color_images:
        assert i1.ndim == 2, 'Images need to be in grayscale according to the IEC standard'
            
    #SNR calculation as defined in 'IEC TS 60904-13':
    signal = 0.5*(i1+i2)-ibg
    noise = 0.5**0.5*np.abs(i1-i2)*((2/np.pi)**-0.5)
    if signal.ndim == 3:#color
        signal = np.average(signal, axis=2, weights=(0.114,0.587,0.299))
        noise = np.average(noise, axis=2, weights=(0.114,0.587,0.299))
    signal = signal.sum()
    noise = noise.sum()
    return signal/noise