import numpy as np
import pickle
import time
from skimage.restoration.deconvolution import unsupervised_wiener, wiener

from imgProcessor.imgIO import imread
from imgProcessor.cameraCalibration.LensDistortion import LensDistortion
from imgProcessor.SingleTimeEffectDetection import SingleTimeEffectDetection
from imgProcessor.cameraCalibration import NoiseLevelFunction
from imgProcessor.filters.medianThreshold import medianThreshold



DATE_FORMAT = "%d %b %y"



def toDate(date):
    if date is None:
        return time.localtime()
    else:
        return time.strptime(date, DATE_FORMAT)

# def newest(dates):
#     return np.argmax([time.mktime(t) for t in dates])


def insertDateIndex(date, l):
    '''
    returns the index to insert the given date in a list
    where each items first value is a date
    '''
    return next((i for i,n in enumerate(l) if n[0] < date), len(l))


def getFromDate(l, date):
    '''
    returns the index of given or best fitting date
    '''        
    if date:
        date = toDate(date)
        i = insertDateIndex(date, l) - 1
        if i == -1:
            return l[0]
        return l[i]
    else:
        return l[0]



class CameraCalibration(object):
    ftype = '.cal'
    
    def __init__(self):
        self.noise_level_function = None
        
        self.coeffs = {
            'name':'no camera',
            'max value':0,       # maximum integer value depending of the bit depth of the camera
            'light spectra':[], # available light spectra e.g. ['light', 'IR']
            'dark current':[],# [ [date, [slope, intercept], info, (error)],[...] ]
            'flat field' : {},# {light:[[date, info, array, (error)],[...] ]
            'lens': {},       # {light:[[ date, info, LensDistortion]         ,[...] ]
            'noise': [],      # [ [date, info, NoiseLevelFunction]            ,[...] ]
            'psf': {},
            'shape':None,
            'balance':{} #factor sharpness/smoothness of image used for wiener deconvolution
            }

    #TODO: rename
    def _getC(self, typ, light):
        d = self.coeffs[typ]
        if type(d) is dict:
            assert light is not None, 'need light spectrum given to access [%s]' %typ
            d = d[light]
        return d        


    def dates(self, typ, light=None):
        try:
            d = self._getC(typ, light)
            return [self.toDateStr(c[0]) for c in d]
        except KeyError:
            return []
  
  
    def infos(self, typ, light=None, date=None):
        d = self._getC(typ, light)        
        if date is None:
            return [c[1] for c in d]
        #TODO: not struct time, but time in ms since epoch 
        return getFromDate(d,date)[1]


    def overview(self):
        '''
        return an overview covering all calibration 
        infos and shapes as string
        '''
        c = self.coeffs
        out = 'camera name: %s' %c['name']
        out+='\nmax value: %s' %c['max value']
        out+='\nlight spectra: %s' %c['light spectra']
        
        out += '\ndark current:'
        for (date, info, (slope, intercept), error) in c['dark current']:
            out += '\n\t date: %s' %self.toDateStr(date)
            out += '\n\t\t info: %s; slope:%s, intercept:%s' %(info, slope.shape, intercept.shape)

        out += '\nflat field:'
        for light, vals in c['flat field'].iteritems():
            out += '\n\t light: %s' %light
            for (date, info, arr, error) in vals:
                out += '\n\t\t date: %s' %self.toDateStr(date)
                out += '\n\t\t\t info: %s; array:%s' %(info, arr.shape)

        out += '\nlens:'
        for light, vals in c['lens'].iteritems():
            out += '\n\t light: %s' %light
            for (date, info, coeffs) in vals:
                out += '\n\t\t date: %s' %self.toDateStr(date)
                out += '\n\t\t\t info: %s; coeffs:%s' %(info, coeffs)

        out += '\nnoise:'
        for (date, info, nlf_coeff, error) in c['noise']:
            out += '\n\t date: %s' %self.toDateStr(date)
            out += '\n\t\t info: %s; coeffs:%s' %(info, nlf_coeff)

        out += '\nPoint spread function:'
        for light, vals in c['psf'].iteritems():
            out += '\n\t light: %s' %light
            for (date, info, psf) in vals:
                out += '\n\t\t date: %s' %self.toDateStr(date)
                out += '\n\t\t\t info: %s; shape:%s' %(info, psf.shape)

        return out
        
    @staticmethod
    def toDateStr(date_struct):
        return time.strftime(DATE_FORMAT, date_struct)
    
    
    @staticmethod        
    def currentTime():
        return time.strftime(DATE_FORMAT)


    def _registerLight(self, light_spectrum):
        if light_spectrum not in self.coeffs['light spectra']:
            self.coeffs['light spectra'].append(light_spectrum)


    def setCamera(self, camera_name, bit_depth=16):
        self.coeffs['name'] = camera_name
        self.coeffs['max value'] = 2**bit_depth-1
        

    def addDarkCurrent(self, slope, intercept, date=None, info='', error=None):
        '''
        slope, intercept, error as array
        slope: dPx/dExposureTime[sec]
        error: absolute
        date: "DD Mon YY" e.g. "30 Nov 16"
        '''
        date = toDate(date)
        
        self._checkShape(slope)
        self._checkShape(intercept)

        d = self.coeffs['dark current']
        d.insert( insertDateIndex(date,d), [date, info, (slope, intercept), error] )


    def addNoise(self, nlf_coeff, date=None, info='', error=None):
        '''
        slope, intercept, error as array
        slope: dPx/dExposureTime[sec]
        error: absolute
        date: "DD Mon YY" e.g. "30 Nov 16"
        '''
        date = toDate(date)
        d = self.coeffs['noise']
        d.insert( insertDateIndex(date,d), [date, info, nlf_coeff, error] )




    def addDeconvolutionBalance(self, balance, date=None, info='', 
                                light_spectrum='visible'):
        self._registerLight(light_spectrum)
        date = toDate(date)

        f = self.coeffs['balance']
        if light_spectrum not in f:
            f[light_spectrum] = []
        f[light_spectrum].insert(insertDateIndex(date,f[light_spectrum]),
                                           [date, info, balance])



    def addPSF(self, psf, date=None, info='', light_spectrum='visible'):
        '''
        add a new point spread function
        '''
        self._registerLight(light_spectrum)
        date = toDate(date)

        f = self.coeffs['psf']
        if light_spectrum not in f:
            f[light_spectrum] = []
        f[light_spectrum].insert(insertDateIndex(date,f[light_spectrum]),
                                           [date, info, psf])
        
        

    def _checkShape(self, array):
        s = self.coeffs['shape']
        if s is None:
            self.coeffs['shape'] = array.shape
        elif s != array.shape:
            raise Exception('array shapes are different: stored(%s), given(%s)' %(s, array.shape))


    def addFlatField(self, arr, date=None, info='', error=None, light_spectrum='visible'):
        '''
        light_spectrum = light, IR ...
        '''
        self._registerLight(light_spectrum)
        self._checkShape(arr)
        date = toDate(date)
        f = self.coeffs['flat field']
        if light_spectrum not in f:
            f[light_spectrum] = []
        f[light_spectrum].insert(insertDateIndex(date,f[light_spectrum]),
                                           [date, info, arr, error])


    def addLens(self, lens, date, info='', light_spectrum='visible'):
        '''
        lens -> instance of LensDistortion or saved file
        '''
        self._registerLight(light_spectrum)
        date = toDate(date)

        if not isinstance(lens, LensDistortion): 
            l = LensDistortion()
            l.readFromFile(lens)
            lens = l

        f = self.coeffs['lens']
        if light_spectrum not in f:
            f[light_spectrum] = []
        f[light_spectrum].insert(insertDateIndex(date,f[light_spectrum]),
                                           [date, info, lens.coeffs])
     

    def clearOldCalibrations(self, date=None):
        '''
        if not only a specific date than remove all except of the youngest calibration
        '''
        self.coeffs['dark current'] = [self.coeffs['dark current'][-1]]
        self.coeffs['noise'] = [self.coeffs['noise'][-1]]
        
        for light in self.coeffs['flat field']:
            self.coeffs['flat field'][light] = [self.coeffs['flat field'][light][-1]]
        for light in self.coeffs['lens']:
            self.coeffs['lens'][light] = [self.coeffs['lens'][light][-1]]


    def _correctPath(self, path):
        if not path.endswith(self.ftype):
            path += self.ftype
        return path


    @staticmethod
    def loadFromFile(path):
        cal = CameraCalibration()
        path = cal._correctPath(path)
        d = pickle.load(open(path,'rb'))
        cal.coeffs.update(d)
        return cal


    def saveToFile(self, path):
        path = self._correctPath(path)
        c = dict(self.coeffs)
        with open(path, 'wb') as outfile:
            pickle.dump(c, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    
    def correct(self,  
                image1,
                image2=None,
                bgImages=None,
                exposure_time=None,
                light_spectrum=None,
                threshold=0.2,
                keep_size=True,
                date=None,
                deblur=True):
        '''
        exposure_time [s]
        
        date -> string e.g. '30. Nov 15' to get a calibration on from date
             -> {'dark current':'30. Nov 15',
                 'flat field':'15. Nov 15',
                 'lens':'14. Nov 15',
                 'noise':'01. Nov 15'}
        '''
        
        if isinstance(date, basestring) or date is None:
            date = {'dark current':date,
                 'flat field':date,
                 'lens':date,
                 'noise':date,
                 'psf':date}
        
        if light_spectrum is None:
            light_spectrum = self.coeffs['light spectra'][0]
            
        assert exposure_time is not None or bgImages is not None,'either exposureTime or bgImages has to be given'

        #0.NOISE
        n = self.coeffs['noise']
        if self.noise_level_function is None and len(n):
            n = getFromDate(n, date['noise'])[2]
            self.noise_level_function = lambda x: NoiseLevelFunction.boundedFunction(x, *n)

        #1. STE REMOVAL ONLY IF 2 IMAGES ARE GIVEN:
        image1orig = image1
        image1 = np.asfarray(imread(image1))
        
        self._checkShape(image1)
        
        if image2 is None:
            image = image1
            if id(image1orig) == id(image):
                image = image.copy()
        else:
            image2 = np.asfarray(imread(image2))
            self._checkShape(image2)
            print('... remove single-time-effects')
            ste = SingleTimeEffectDetection((image1,image2), nStd=4, 
                            noise_level_function=self.noise_level_function)
            image = ste.noSTE
            if self.noise_level_function is None: 
                self.noise_level_function = ste.noise_level_function
            
        self.last_light_spectrum = light_spectrum
        self.last_img = image
        #2. BACKGROUND REMOVAL
        print('... remove background')
        self._correctDarkCurrent(image, exposure_time, bgImages, 
                                        date['dark current'])

        #3. VIGNETTING/SENSITIVITY CORRECTION:
        print('... remove vignetting and sensitivity')
        self._correctVignetting(image, light_spectrum, 
                                       date['flat field'])

        #4. REPLACE DECECTIVE PX WITH MEDIAN FILTERED FALUE
        print('... remove artefacts')
        self._correctArtefacts(image, threshold)

        #5. DEBLUR
        if deblur:
            print('... remove blur')
            image = self._correctBlur(image, light_spectrum, date['psf'])
        #5. LENS CORRECTION:
        print('... correct lens distortion')
        image = self._correctLens(image, light_spectrum, date['lens'], 
                                 keep_size)

        print('DONE')
        return image


    def _correctDarkCurrent(self, image, exposuretime, bgImages, date):
        '''
        open OR calculate a background image: f(t)=m*t+n
        '''
        if bgImages is not None:
            if ( type(bgImages) in (list, tuple) or 
                (isinstance(bgImages, np.ndarray) and bgImages.ndim==3) ):
                #if multiple images are given: do STE removal:
                bg = SingleTimeEffectDetection(
                    (imread(bgImages[0]),imread(bgImages[1])), nStd=4).noSTE
            else:
                bg = imread(bgImages)
        else:
            d = self.coeffs['dark current']
            d = getFromDate(d, date)
            #calculate bg image:
            offs,ascent = d[2]
            bg = offs + ascent*exposuretime
            mx = self.coeffs['max value']
            with np.errstate(invalid='ignore'):
                bg[bg>mx] = mx    
        image-=bg


    def _correctVignetting(self, image, light_spectrum, date):
        d = self.getCoeff('flat field', light_spectrum, date)[2]
        i = d!=0
        image[i]/=d[i]
#         with np.errstate(divide='ignore'):
#             out = image / d
#         #set 
#         
#         out[i]=image[i]
        #return image



    def _correctBlur(self, image, light_spectrum, date):
        d = self.getCoeff('psf', light_spectrum, date)
        if not d:
            print 'skip deconvolution // no PSF set'
            return image
        psf = d[2]
        mx = image.max()
        image/=mx
        
        balance = self.getCoeff('balance', light_spectrum, date)
        if balance is None:
            print 'no balance value for wiener deconvolution found // use unsupervised_wiener instead // this will take some time'
            deconvolved, _ = unsupervised_wiener(image, psf)
        else:
            deconvolved = wiener(image, psf, balance[2])
        deconvolved[deconvolved<0]=0
        deconvolved*=mx
        return deconvolved
   
    
    def _correctArtefacts(self, image, threshold):
        '''
        Apply a thresholded median replacing high gradients 
        and values beyond the boundaries
        '''
        medianThreshold(image, threshold, copy=False)


    def getLens(self, light_spectrum, date):
        d = self.getCoeff('lens', light_spectrum, date)[2]
        return LensDistortion(d)        


    def _correctLens(self, image, light_spectrum, date, keep_size):
        lens = self.getLens(light_spectrum, date)
        return lens.correct(image, keepSize=keep_size) 
  

    def deleteCoeff(self, name, date, light=None):
        try:
            c = self.coeffs[name][light]
        except TypeError:
            #not light dependent
            c = self.coeffs[name]
        d = toDate(date)
        i = insertDateIndex(d,c) - 1
        if i != -1:
            c.pop(i)
        else:
            raise Exception('no coeff %s for date %s' %(name, date))


    def getCoeff(self, name, light, date):
        '''
        try to get calibration for right light source, but
        use another if they is none existent
        '''
        d = self.coeffs[name]
        
        try:
            c= d[light]
        except KeyError:
            try:
                k,i = d.iteritems().next()
                print('no calibration found for [%s] - using [%s] instead' %(light, k))
            except StopIteration:
                return None
            c = i
        except TypeError:
            #coeff not dependent on light source
            c = d
        return getFromDate(c, date)


    def uncertainty(self, img=None, light_spectrum=None):
        #TODO: review
        if img is None:
            img = self.last_img
        if light_spectrum is None:
            light_spectrum = self.last_light_spectrum

        s = img.shape
        position = self.coeffs['lenses'][light_spectrum].getUncertainty(s[1],s[0])

        intensity = (
                     self.coeffs['dark_RMSE']**2 +
                    (self.coeffs['vignetting_relUncert'][light_spectrum]*img)**2 +
                    self.coeffs['sensitivity_RMSE']**2
                    )**0.5
        #make relative:
        img = img.copy()
        img[img==0]=1
        intensity /= img
        #apply lens distortuion:
        intensity = self.coeffs['lenses'][light_spectrum].correct(
                                            intensity,keepSize=True) 
        return intensity, position



if __name__ == '__main__':
    #TODO: generate synthetic distortion img and calibration
    pass
#     from fancytools.os.PathStr import PathStr
#     import imgProcessor
#     c = CameraCalibration.loadFromFile(
#             'C:\\Users\\elkb4\\Desktop\\PhD\\Measurements\\cameraCalibrations\\HuLC5.cal')
#     print c.overview()
# 
# 
#     p = PathStr('C:\Users\elkb4\Desktop\PhD\Measurements\HuLC\erik_storm_damage\\0275\\100')
#     i1 = p.join('erik_0275_e30000_g4_b1_V42-029_I9-432_T19-062_p1-2_n1__1.tif')
#     i2 = p.join('erik_0275_e30000_g4_b1_V42-143_I9-432_T19-062_p1-2_n0__0.tif')
#     
#     imgProcessor.ARRAYS_ORDER_IS_XY = True
#     
#     i3 = c.correct(i1,i2,
#                 exposure_time=30,
#                 threshold=0.2,
#                 keep_size=True,
#                 date=None,
#                 deblur=True)
#     print i3.shape
#     print i3
#     import pylab as plt
#     plt.imshow(i3)
#     plt.show()
#     np.save('corrected', i3)
#     
