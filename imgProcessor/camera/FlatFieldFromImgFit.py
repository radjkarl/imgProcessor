import numpy as np

from scipy.ndimage.filters import gaussian_filter, maximum_filter, \
                                  laplace, minimum_filter
from skimage.transform import resize

from fancytools.math.MaskedMovingAverage import MaskedMovingAverage
from fancytools.fit.fit2dArrayToFn import fit2dArrayToFn

from imgProcessor.imgIO import imread
from imgProcessor.measure.FitHistogramPeaks import FitHistogramPeaks
from imgProcessor.signal import getSignalPeak
from imgProcessor.equations.vignetting import vignetting
from imgProcessor.interpolate.polyfit2d import polyfit2dGrid



def highGrad(arr):
    #mask high gradient areas in given array 
    s = min(arr.shape)
    return maximum_filter(np.abs(laplace(arr, mode='reflect')) > 0.02,
                          min(max(s/5,3),15) )



class FlatFieldFromImgFit(object):
    def __init__(self, images=None, nstd=3, ksize=None, scale_factor=None):
        '''
        calculate flat field from multiple non-calibration images
        through ....
        * blurring each image
        * masked moving average of all images to even out individual deviations
        * fit vignetting function of average OR 2d-polynomal
        '''
        self.nstd = nstd
        self.ksize = ksize
        self.scale_factor = scale_factor
        
        self.bglevel = 0 #average background level
        self._mx = 0
        self._n = 0
        self._m = None
        if images is not None:
            for n,i in enumerate(images):
                print '%s/%s' %(n+1,len(images))
                self.addImg(i)
            
            
    def addImg(self, i):
            img = imread(i, 'gray', dtype=float)
            self._orig_shape = img.shape

            if self.scale_factor is None:
                #determine so that smaller image size has 50 px
                self.scale_factor = 100.0/min(img.shape)
            s = [int(s*self.scale_factor) for s in img.shape]
 
            img = resize(img,s)

            if self._m is None:           
                self._m = MaskedMovingAverage(shape=img.shape)
                if self.ksize is None:
                    self.ksize = max(3, int(min(img.shape)/10))

            f = FitHistogramPeaks(img)
            sp  = getSignalPeak(f.fitParams)
            
            #non-backround indices:
            ind = img > sp[1]-self.nstd*sp[2]
            #blur:
            blurred = minimum_filter(img,3)
            blurred = maximum_filter(blurred,self.ksize)
            gblurred = gaussian_filter(blurred, self.ksize)
            blurred[ind]=gblurred[ind]

            #scale [0-1]:
            mn = img[~ind].mean()
            if np.isnan(mn):
                mn = 0
            mx = blurred.max()
            blurred-=mn
            blurred/=(mx-mn)
            
            ind = blurred>self._m.avg
            
            self._m.update(blurred, ind)
            self.bglevel += mn
            self._mx += mx    

            self._n +=1


    def flatFieldFromFunction(self): 
        '''
        calculate flatField from fitting vignetting function to averaged fit-image
        returns flatField, average background level, fitted image, valid indices mask
        '''   
        s0,s1 = self._m.avg.shape
                #f-value, alpha, fx, cx,     cy
        guess = (s1*0.7,  0,     1 , s0/2.0, s1/2.0)
        
        #set assume normal plane - no tilt and rotation:
        fn = lambda (x,y),f,alpha, fx,cx,cy:  vignetting((x*fx,y),  f, alpha, 
                cx=cx,cy=cy)
    
        fitimg = self._m.avg
        mask = fitimg>0.5
        
        flatfield = fit2dArrayToFn(fitimg, fn, mask=mask, 
                        guess=guess,output_shape=self._orig_shape)[0]
        
        return flatfield, self.bglevel/self._n, fitimg, mask



    def flatFieldFromFit(self):
        '''
        calculate flatField from 2d-polynomal fit filling
        all high gradient areas within averaged fit-image
        
        returns flatField, average background level, fitted image, valid indices mask
        '''
        
        fitimg = self._m.avg
        #replace all dark and high gradient variations:
        mask = np.logical_or(fitimg < 0.5, highGrad(fitimg))  
  
        out = fitimg.copy()
        lastm = 0

        for _ in xrange(10):
            out = polyfit2dGrid(out, mask, 2)
            mask =  highGrad(out) 
            m = mask.sum()
            if m == lastm:
                break
            lastm = m

        out = np.clip(out,0.1,1) 

        out = resize(out,self._orig_shape, mode='reflect')
        return  out, self.bglevel / self._n, fitimg, mask



if __name__ == '__main__':
    import pylab as plt
    import sys
    
    s0,s1 = 200,300
            #f-value, alpha,  rot, tilt,cx,     cy
    params = (s1*0.7,  0,     0,   0 ,  s0/2.0, s1/2.0)
    vig = np.fromfunction(lambda x,y: vignetting((x,y),*params),  (s0,s1))


    o0,o1 = 120,150
    p0,p1 = 50,75
    d0,d1 = 10,15

    ff = FlatFieldFromImgFit()
    
    #lets say we have 10 images of an object at slightly different positions
    for c in xrange(10):
        img = np.zeros((s0,s1))
        dev0 = np.random.rand()*d0
        dev1 = np.random.rand()*d1
        img[dev0+p0:dev0+p0+o0,dev1+p1:dev1+p1+o1] = 1
        img += np.random.rand(s0,s1)
        img *=vig
        ff.addImg(img)
        print '%i/%i' %(c,10)
        
    vig_fit = ff.flatFieldFromFit()[0]
    vig_fn,bg,img_fit,mask = ff.flatFieldFromFunction()
    print('background level: %s' %bg)

    if 'no_window' not in sys.argv:
        plt.figure('image 10/10')
        plt.imshow(img)
        plt.colorbar()
    
        plt.figure('original flat field')
        plt.imshow(vig)
        plt.colorbar()
    
        plt.figure('fitted flat field / function')
        plt.imshow(vig_fn)
        plt.colorbar()
    
        plt.figure('fitted flat field / polynomal fit')
        plt.imshow(vig_fit)
        plt.colorbar()
    
        plt.figure('image average')
        plt.imshow(img_fit)
        plt.colorbar()
    
        plt.figure('valid indices')
        plt.imshow(mask)
        plt.colorbar()
    
        plt.show()
        
        
        
