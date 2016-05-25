import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize.minpack import curve_fit



class SharpnessBase(object):
    '''
    Base class for all classes to determinate the 
    Point-spread-function
    '''
    
    def __init__(self):
        self._fitParam = None
        self._corrPsf = None #corrected PSF
        self._std = None


    def MTF50(self, MTFx,MTFy):
        '''
        return object resolution as [line pairs/mm]
               where MTF=50%
               see http://www.imatest.com/docs/sharpness/
        '''
        if self.mtf_x is None:
            self.MTF()
        f = UnivariateSpline(self.mtf_x, self.mtf_y-0.5)
        return f.roots()[0]
    

    def MTF(self, px_per_mm):
        '''
        px_per_mm = cam_resolution / image_size
        '''
        res = 100 #numeric resolution
        r = 4 #range +-r*std
        
        #size of 1 px:
        px_size = 1./px_per_mm
        
        #standard deviation of the point-spread-function (PSF) as normal distributed:
        std = self.std*px_size #transform standard deviation from [px] to [mm]

        x = np.linspace(-r*std,r*std, res)
        #line spread function:
        lsf = self.gaussian1d(x, 1, 0, std)
        #MTF defined as Fourier transform of the line spread function:
            #abs() because result is complex
        y = abs(np.fft.fft(lsf)) 
            #normalize fft so that max = 1
        y /= np.max(y)
            #step length between xn and xn+1
        dstep = r*std/res
            # Fourier frequencies - here: line pairs(cycles) per mm
        freq = np.fft.fftfreq(lsf.size, dstep)
        #limit mtf between [0-px_per_mm]:
        i = np.argmax(freq>px_per_mm)
        self.mtf_x = freq[:i]
        self.mtf_y = y[:i]
        return self.mtf_x, self.mtf_y


    #TODO: review
    def uncertaintyMap(self, psf, method='convolve', fitParams=None):
        '''
        return the intensity based uncertainty due to the unsharpness of the image
        as standard deviation
        
        method = ['convolve' , 'unsupervised_wiener']
                    latter one also returns the reconstructed image (deconvolution)
        '''

        #ignore background:
        #img[img<0]=0
        ###noise should not influence sharpness uncertainty:
        ##img = median_filter(img, 3)

        # decrease noise in order not to overestimate result:
        img = scaleSignal(self.img, fitParams=fitParams)

        if method == 'convolve':
            #print 'convolve'
            blurred = convolve2d(img, psf, 'same')
            m = abs(img-blurred) / abs(img + blurred)
            m = np.nan_to_num(m)
            m*=self.std**2
            m[m>1]=1
            self.blur_distortion = m
            np.save('blurred', blurred)
            return m
        else:
            restored = unsupervised_wiener(img, psf)[0]
            m = abs(img-restored) / abs(img + restored)
            m = np.nan_to_num(m)
            m*=self.std**2
            m[m>1]=1
            self.blur_distortion = m
            return m, restored
# 
#         #FOLLOWING DIDNT WORK OUT ... DELETE?? 
#         sy,sx = img.shape
#         
#         #create following objects now so that the jit compiler can handle the following fn:
#             #expected value (http://en.wikipedia.org/wiki/Variance#Discrete_random_variable)
#         self.umap_mue = np.zeros(shape=(sy,sx), dtype=np.float64)#self.img.dtype)
#             #standard deviation
#         self.umap_std = np.zeros(shape=(sy,sx), dtype=np.float64)#self.img.dtype)
#         mx,my = psf.shape
#         buff = np.empty(shape=psf.shape).astype(np.float64)#self.img.dtype)
#         
#         @jit(nopython=True) #nogil=True)
#         def calc(sx, sy, my, mx, img, pdf,umap_mue, umap_std, buff, av_std):
#             mmx = mx/2
#             mmy = my/2 
#             for py in xrange(mmy,sy-mmy-1):
#                 for px in xrange(mmx,sx-mmx-1):
#                     #for every pixel in the scaled image:
#                     #take a snipped of size fineness*fineness
#                     snipped = img[py-mmy:py+mmy+1,px-mmx:px+mmx+1]
#                     #subtract the vsalue in the middle (=get the changes regaring the 
#                     #current pixel:
#                     np.subtract(snipped,snipped[mmy,mmx], buff)
#                     mue = 0.0
#                     std = 0.0
#                     #calculate expected value mue:
#                     for i in range(my):
#                         for j in range(mx):
#                             mue += buff[i,j]*pdf[i,j]
#                     #calculate variance (http://en.wikipedia.org/wiki/Variance#Discrete_random_variable)
#                     for i in range(my):
#                         for j in range(mx):
#                             std += pdf[i,j]*(av_std**2*(buff[i,j]-mue))**2
#                     umap_mue[py,px] = mue
#                     umap_std[py,px] = std**0.5
#                     
#         calc(sx, sy, my, mx, img, psf,self.umap_mue, self.umap_std, buff, self.std)
# 
#         self.blur_distortion = self.umap_std + np.abs(self.umap_mue)
# 
#         return self.blur_distortion

  
    def _psfGridCoords(self):
        s = self._corrPsf.shape
        x,y = np.mgrid[0:s[0],0:s[1]]
        x = (x-s[0]//2)
        y = (y-s[1]//2)
        return x,y


    def gaussianPsf(self):
        if self._fitParam is None:
            self.stdDev()
        x,y = self._psfGridCoords()
        return self._fn((x,y), *self._fitParam)


    def stdDev(self):
        '''
        get the standard deviation 
        from the PSF is evaluated as 2d Gaussian
        '''
        if self._corrPsf is None:
            self.psf()
        p = self._corrPsf.copy()
        mn = p.min()
        p[p<0.05*p.max()] = mn
        p-=mn
        p/=p.sum()
        
        x,y = self._psfGridCoords()
        x = x.flatten()
        y = y.flatten()

        guess = (1,1,0)

        param, _ = curve_fit(self._fn, (x,y), p.flatten(), guess)

        self._fitParam = param 
        stdx,stdy =  param[:2]
        self._std = (stdx+stdy)/2
        
        return self._std