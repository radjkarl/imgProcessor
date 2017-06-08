import numpy as np


def addNoise(img, snr=25, rShot=0.5):
    '''
    adds Gaussian (thermal) and shot noise to [img]

    [img] is assumed to be noise free

    [rShot] - shot noise ratio relative to all noise 
    '''
    s0, s1 = img.shape[:2]

    m = img.mean()
    if np.isnan(m):
        m = np.nanmean(img)
    assert m != 0, 'image mean cannot be zero'

    img = img / m
    noise = np.random.normal(size=s0 * s1).reshape(s0, s1)
    if rShot > 0:
        noise *= (rShot * img**0.5 + 1)
    noise /= np.nanstd(noise)
    noise[np.isnan(noise)] = 0
    return m * (img + noise / snr)


if __name__ == '__main__':
    import pylab as plt
    import sys
    from imgProcessor.generate.patterns import patCircles
    from imgProcessor.measure.SNR.SNR_IEC import SNR_IEC

    SNR = 15

    img = patCircles(100)
    i1 = addNoise(img, snr=SNR)
    i2 = addNoise(img, snr=SNR)

    # TEST WHETHER ADDED NOISE CAUSES CORRECT SNR:
    assert abs(SNR_IEC(i1, i2) - SNR) < 1

    if 'no_window' not in sys.argv:
        plt.imshow(i1)
        plt.show()
