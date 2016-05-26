import numpy as np

from fancytools.math.MaskedMovingAverage import MaskedMovingAverage

from imgProcessor.camera.NoiseLevelFunction import oneImageNLF
from imgProcessor.camera.DarkCurrentMap import averageSameExpTimes
from imgProcessor.imgIO import imread


    
def flatFieldFromCalibration(bgImages, images, calcStd=False):
    '''
    returns a flat-field correction map 
    through conditional average of multiple images reduced by a background image
    
    calcStd -> set to True to also return the standard deviation
    '''
    #AVERAGE BACKGROUND IMAGES IF MULTIPLE ARE GIVEN:
    if ( type(bgImages) in (tuple, list) 
            or type(bgImages) is np.ndarray and bgImages.ndim == 3 ) :
        if len(bgImages) > 1:
            avgBg = averageSameExpTimes(bgImages)
        else:
            avgBg = imread(bgImages[0])
    else:
        avgBg = imread(bgImages)

    i0 = imread(images[0]) - avgBg
    noise_level_function,_ = oneImageNLF(i0)

    m = MaskedMovingAverage(shape=i0.shape, calcVariance=calcStd)
    m.update(i0)
    
    for i in images[1:]:
        i = imread(i)
        thresh = m.avg - noise_level_function(m.avg) * 3
        m.update(i, i>thresh)

    mx = m.avg.max()
    if calcStd:
        return m.avg/mx, m.var**0.5/mx
    return m.avg/mx



if __name__ == '__main__':
    from imgProcessor.equations.vignetting import vignetting
    from matplotlib import pyplot as plt
    import sys

    #make 10 vignetting arrays with slightly different optical centre
    #to simulate effects that occur when vignetting is measured badly
    d = np.linspace(-20,20,10)
    bg = np.random.rand(100,100)*10
    vigs = [np.fromfunction(lambda x,y: 
                vignetting((x,y), cx=50-di, cy=50+di),(100,100))*100+bg for di in d] 
    
    
    avg, std = flatFieldFromCalibration(bg, vigs, calcStd=True)

    if 'no_window' not in sys.argv:
        plt.figure('example vignetting img (1/10)')
        plt.imshow(vigs[0])
        plt.colorbar()
    
        plt.figure('example vignetting img (10/10)')
        plt.imshow(vigs[-1])
        plt.colorbar()
        
        plt.figure('averaged vignetting array')
        plt.imshow(avg)   
        plt.colorbar()
    
        plt.figure('standard deviation')
        plt.imshow(std)  
        plt.colorbar()
    
        plt.show()
    
    
