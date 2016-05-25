from scipy.ndimage.measurements import label

import numpy as np

from imgProcessor.cameraCalibration.NoiseLevelFunction import oneImageNLF
from imgProcessor.filters.removeSinglePixels import removeSinglePixels
from imgProcessor.imgIO import imread



class SingleTimeEffectDetection(object):
    '''
    Detect and remove Single-time-effects (STE) using min. 2 equivalent images
    public attributes:
    
    .mask_clean --> STE-free indices 
    .mask_STE   --> STE indices (only avail. if save_ste_indices=True)
    .noSTE      --> STE free image
    '''
    def __init__(self, images, noise_level_function=None, nStd=4, 
                 save_ste_indices=False):
        self.save_ste_indices = save_ste_indices
        self.mask_STE = None
        
        i1 = imread(images[0], 'gray')
        i2 = imread(images[1], 'gray')
        
        if save_ste_indices:
            self.mask_STE = np.zeros(shape=i1.shape, dtype=bool)
        
        #MINIMUM OF BOTH IMAGES:  
        self.noSTE = m = np.min((i1,i2),axis=0)
        if noise_level_function is None:
            noise_level_function = oneImageNLF(m)[0]
        self.noise_level_function = noise_level_function        
        self.threshold = noise_level_function(m)*nStd
        
        self._c = 2
        self.addImage(np.max((i1,i2),axis=0))

        for i in images[2:]:
            self.addImage(imread(i, 'gray'))

          
    def addImage(self, image):
        self._last_diff = diff = image-self.noSTE
        
        ste = diff > self.threshold  
        removeSinglePixels(ste)
        self.mask_clean = clean = ~ste
        self.noSTE[clean] += diff[clean] / self._c
        self._c += 1

        if self.save_ste_indices:
            self.mask_STE += ste
        return self


    def countSTE(self):
        ''' 
        return number of found STE
        '''
        return label(self.mask_STE)[1]
    
    
    def relativeAreaSTE(self):
        '''
        return STE area - relative to image area
        '''
        s = self.noSTE.shape
        return float(np.sum(self.mask_STE)) / (s[0]*s[1])


    def intensityDistributionSTE(self, bins=10, range=None):
        '''
        return distribution of STE intensity
        '''
        v = np.abs(self._last_diff[self.mask_STE])
        return np.histogram(v, bins, range)