import numpy as np

from collections import OrderedDict

from fancytools.math.MaskedMovingAverage import MaskedMovingAverage
from fancytools.math.linRegressUsingMasked2dArrays import linRegressUsingMasked2dArrays

from imgProcessor.imgIO import imread
from imgProcessor.features.SingleTimeEffectDetection import SingleTimeEffectDetection
from imgProcessor.utils.baseClasses import Iteratives



class DarkCurrentMap(Iteratives):
    '''
    Averages given background images
    removing single time effects
    '''
    
    def __init__(self, twoImages, noise_level_function=None, 
                 calcVariance=False, **kwargs):
        Iteratives.__init__(self, **kwargs)
        
        assert len(twoImages) > 1, 'need at least 2 images'

        self.det = SingleTimeEffectDetection(twoImages, noise_level_function, nStd=3)
        self._map = MaskedMovingAverage(shape=twoImages[0].shape, calcVariance=calcVariance)
        self._map.update(self.det.noSTE)


    def addImg(self, img, raiseIfConvergence=False):
        self._map.update( img, self.det.addImage(img).mask_clean )
        if raiseIfConvergence:
            return self.checkConvergence(self.fine.var**0.5)


    def map(self):
        return self._map.avg
    
    
    def uncertaintyMap(self):
        return self._map.var**0.5


    def uncertainty(self):
        return np.mean(self._map.var)**0.5



def averageSameExpTimes(imgs_path):
    '''
    average background images with same exposure time
    '''
    firsts = imgs_path[:2]
    imgs = imgs_path[2:]
    for n, i in enumerate(firsts):
        firsts[n] = np.asfarray(imread(i))
    d = DarkCurrentMap(firsts)
    for i in imgs:
        i = imread(i)
        d.addImg(i)
    return d.map()


def getLinearityFunction(expTimes, imgs, mxIntensity=65535,min_ascent=0.001,
                         #maxExpTime=36000
                         ):
    '''
    returns offset, ascent 
    of image(expTime) = offset + ascent*expTime
    '''
    ascent, offset, error = linRegressUsingMasked2dArrays(expTimes, imgs, imgs > mxIntensity)
    
    ascent[np.isnan(ascent)] = 0

    #TODO: calculate min ascent from noise function
    #remove low frequent noise:
    if min_ascent > 0:
        i = ascent<min_ascent
        #last - worked, but next seem to be better
        offset[i] += (0.5*( np.min(expTimes)+ np.max(expTimes)) )*ascent[i]
        ascent[i] = 0

    #TODO: REMOVE??#########
#     maxExpTime_ = (mxIntensity-offset)/ascent
#     maxExpTime_[maxExpTime_<0]=0
#     maxExpTime_[maxExpTime_>maxExpTime]=0
    return offset, ascent, error#, maxExpTime_


def sortForSameExpTime(expTimes, img_paths, excludeSingleImg=True):
    '''
    return image paths sorted for same exposure time
    '''
    d = {}
    for e,i in zip(expTimes, img_paths):
        if e not in d:
            d[e] = []
        d[e].append(i)
    for key in d.keys():
        if len(d[key]) == 1:
            print 'have only one image of exposure time %s' %key
            print 'excluding that one'
            d.pop(key)
    d = OrderedDict(sorted(d.items()))
    return d.keys(), d.values()


def getDarkCurrentAverages(exposuretimes, imgs):
    '''
    return exposure times, image averages for each exposure time
    '''
    x, imgs_p = sortForSameExpTime(exposuretimes, imgs)
    imgs = []
    for i in imgs_p:
        imgs.append(averageSameExpTimes(i))
    return x, imgs


def getDarkCurrentFunction(exposuretimes, imgs, **kwargs):
    '''
    get dark current function from given images and exposure times
    '''
    exposuretimes, imgs = getDarkCurrentAverages(exposuretimes, imgs)
    offs, ascent, rmse = getLinearityFunction(exposuretimes, np.array(imgs), **kwargs) 
    return offs, ascent, rmse



if __name__ == '__main__':
    import pylab as plt
    import sys
    
    #generate some random images for the following exposure times:
    exposuretimes = range(10,100,20)*3
    print 'exposure times:, ', exposuretimes
    offs = np.random.randint(0,100,(30,100))
    ascent = np.random.randint(0,10,(30,100))
    noise = lambda: np.random.randint(0,10,(30,100))
    #calculate every image as function of exposure time
    #and add noise:
    imgs = [offs+t*ascent+noise() for t in exposuretimes]
    
    offs2,ascent2,rmse = getDarkCurrentFunction(exposuretimes, imgs)

    if 'no_window' not in sys.argv:
        plt.figure("image 1")
        plt.imshow(imgs[1])
        plt.colorbar()
    
        plt.figure("image 5")
        plt.imshow(imgs[5])    
        plt.colorbar()
    
        plt.figure("calculated image 1")
        plt.imshow(offs2+exposuretimes[1]*ascent2)
        plt.colorbar()
    
        plt.figure("calculated image 5")
        plt.imshow(offs2+exposuretimes[5]*ascent2) 
        plt.colorbar()
    
        plt.show()