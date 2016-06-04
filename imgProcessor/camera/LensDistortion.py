#!/usr/bin/env python
import numpy as np
import cv2
from collections import OrderedDict

#own
import imgProcessor
from imgProcessor.imgIO import imread
from imgProcessor.exceptions import NothingFound, EnoughImages



class LensDistortion(object):
    '''
    class for detecting and correcting the lens distortion within images
    
    based on:
    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    '''
    ftype = 'npz'

    def __init__(self, coeffs={}):
        #SAVED CALIBRATION RESULTS
        self._coeffs = coeffs
        #SAVED CALIBRATION SETTINGS
        self.opts = {}
        self.mapx, self.mapy = None, None


    def calibrate(self, board_size, method='Chessboard', images=[], 
                        max_images=100,sensorSize_mm=None,
                        detect_sensible=False):
        '''
        sensorSize_mm - (width, height) [mm] Physical size of the sensor
        '''        
        self._coeffs = {}
        self.opts = {'foundPattern':[],#whether pattern could be found for image
                     'size':board_size,
                     'imgs':[], #list of either npArrsays or img paths
                     'imgPoints':[] # list or 2d coords. of found pattern features (e.g. chessboard corners)
                     }
                                      

        self._detect_sensible = detect_sensible

        self.method = {'Chessboard':self._findChessboard,
                       'Symmetric circles':self._findSymmetricCircles, 
                       'Asymmetric circles':self._findAsymmetricCircles,
                       'Manual':None
                       #TODO: 'Image grid':FindGridInImage
                       }[method]
        
        self.max_images = max_images
        self.findCount = 0
        self.apertureSize = sensorSize_mm
        
        s0,s1 = board_size
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((s0*s1,3), np.float32)
            #reshape 3-D matrix as 2-D
        self.objp[:,:2] = np.mgrid[0:s0,0:s1].T.reshape(-1,2)
        
        if method == 'Asymmetric circles':
            #this pattern have its points (every 2. row) displaced, so:
            i = self.objp[:,1]%2==1
            self.objp[:,0]*=2
            self.objp[i,0]+=1
        
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        #self.imgpoints = [] # 2d points in image plane.
        self.mapx, self.mapy = None, None
        
        #from matplotlib import pyplot as plt
        for n,i in enumerate(images):
            print('working on image %s' %n)
            if self.addImg(i):
                print('OK')
        
        
    def addPoints(self, points):
        '''
        add corner points directly instead of extracting them from 
        image
        points = ( (0,1), (...),... ) [x,y]
        '''
        self.opts['foundPattern'].append(True)
        self.findCount += 1
        self.objpoints.append(self.objp)
        s0 = points.shape[0]
        
        self.opts['imgPoints'].append(np.asarray(points
                                                 ).reshape(s0,1,2
                                                 ).astype(np.float32)) 
    def setImgShape(self, shape):
        '''
        image shape must be known for calculating camera matrix
        if method==Manual and addPoints is used instead of addImg
        this method must be called before .coeffs are obtained
        '''
        self.img = type('Dummy', (object,), {}) 
        if imgProcessor.ARRAYS_ORDER_IS_XY:
            self.img.shape = shape[::-1]
        else:
            self.img.shape = shape

   
    def addImgStream(self, img):
        '''
        add images using a continous stream 
        - stop when max number of images is reached
        '''
        if self.findCount > self.max_images:
            raise EnoughImages('have enough images') 
        return self.addImg(img)


    def addImg(self, img):
        '''
        add one chessboard image for detection lens distortion
        '''
        #self.opts['imgs'].append(img)

        self.img = imread(img, 'gray', 'uint8')

        didFindCorners, corners = self.method()
        self.opts['foundPattern'].append(didFindCorners)
        
        if didFindCorners:
            self.findCount += 1
            self.objpoints.append(self.objp)
            self.opts['imgPoints'].append(corners) 
        return didFindCorners


    def _findChessboard(self):
        # Find the chess board corners
        flags = cv2.CALIB_CB_FAST_CHECK
        if self._detect_sensible:
            flags = (cv2.CALIB_CB_FAST_CHECK |
                     cv2.CALIB_CB_ADAPTIVE_THRESH |
                     cv2.CALIB_CB_FILTER_QUADS |
                     cv2.CALIB_CB_NORMALIZE_IMAGE )
            
        (didFindCorners, corners) = cv2.findChessboardCorners(
                self.img, self.opts['size'], flags=flags
                    )
        if didFindCorners:
            #further refine corners, corners is updatd in place
            cv2.cornerSubPix(self.img,corners,(11,11),(-1,-1), 
                             # termination criteria for corner estimation for chessboard method
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                              30, 0.001)
                            ) # returns None
        return didFindCorners, corners


    def _findAsymmetricCircles(self):
        return self._findSymmetricCircles(flags=cv2.CALIB_CB_ASYMMETRIC_GRID)


    def _findSymmetricCircles(self, flags=cv2.CALIB_CB_SYMMETRIC_GRID):
        (didFindCorners, corners) = cv2.findCirclesGridDefault(
                self.img, self.opts['size'], 
                flags=flags|cv2.CALIB_CB_CLUSTERING)
        return didFindCorners, corners


    def getCoeffStr(self):
        '''
        get the distortion coeffs in a formated string 
        '''
        txt = ''
        for key, val in self.coeffs.iteritems():
            txt += '%s = %s\n' %(key, val)
        return txt


    def drawChessboard(self, img=None):
        '''
        draw a grid fitting to the last added image 
        on this one or an extra image
        img == None
            ==False -> draw chessbord on empty image
            ==img
        ''' 
        assert self.findCount > 0, 'cannot draw chessboard if nothing found'
        if img is None:
            img = self.img
        elif type(img) == bool and img == False:
            img = np.zeros(shape=(self.img.shape), dtype=self.img.dtype)
        else:
            img = imread(img, dtype='uint8')
        gray = False  
        if img.ndim == 2:
            gray=True
            #need a color 8 bit image
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, self.opts['size'], 
                                  self.opts['imgPoints'][-1], 
                                  self.opts['foundPattern'][-1])
        if gray:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        return img


    @property
    def coeffs(self):
        if not self._coeffs:
            if not self.findCount:
                raise NothingFound('can create camera calibration because no corners have been found')
            
            #http://en.wikipedia.org/wiki/Reprojection_error
            try:
                (reprojectionError, cameraMatrix, distortionCoeffs, 
                 rotationVecs, translationVecs) = cv2.calibrateCamera(
                            self.objpoints, 
                            self.opts['imgPoints'], 
                            self.img.shape[::-1], None, None)
                print('reprojectionError=%s' %reprojectionError)
            except Exception, err:
                raise NothingFound(err)
            
            self._coeffs = OrderedDict([
                    ('reprojectionError',reprojectionError),
                    ('apertureSize',self.apertureSize),
                    ('cameraMatrix',cameraMatrix),
                    ('distortionCoeffs',distortionCoeffs),
                    ('shape', self.img.shape),
                    #('rotationVecs',rotationVecs),
                    #('translationVecs',translationVecs),
                    ])
            if self.apertureSize is not None:
                (fovx, fovy, focalLength, principalPoint, 
                 aspectRatio) = cv2.calibrationMatrixValues(
                        cameraMatrix, self.img.shape, *self.apertureSize)
                self._coeffs.update(OrderedDict([
                    ('fovx',fovx),
                    ('fovy',fovy),
                    ('focalLength',focalLength),
                    ('principalPoint',principalPoint),
                    ('aspectRatio',aspectRatio)]) 
                    )
        return self._coeffs
    
    
    @coeffs.setter
    def coeffs(self, c):
        self._coeffs = c


    def writeToFile(self, filename, saveOpts=False):
        '''
        write the distortion coeffs to file
        saveOpts --> Whether so save calibration options (and not just results)
        '''
        try:
            if not filename.endswith('.%s' %self.ftype):
                filename += '.%s' %self.ftype
            s = {'coeffs': self.coeffs}
            if saveOpts:
                s['opts'] = self.opts
#             else:
#                 s['opts':{}]
            np.savez(filename, **s)
            return filename
        except AttributeError:
            raise Exception('need to calibrate camera before calibration can be saved to file')


    def readFromFile(self, filename):
        '''
        read the distortion coeffs from file
        '''
        s = dict(np.load(filename))
        self.coeffs = s['coeffs'][()]
        try:
            self.opts = s['opts'][()]
        except KeyError:
            pass
        return self.coeffs



    def undistortPoints(self, points):
        s = self.img.shape
        cam = self.coeffs['cameraMatrix']
        d = self.coeffs['distortionCoeffs']
 
        (newCameraMatrix, _) = cv2.getOptimalNewCameraMatrix(cam, 
                                    d, (s[1], s[0]), 1, 
                                    (s[1], s[0]))
        return cv2.undistortPoints(points,cam, d, P=newCameraMatrix)


    def correct(self, image, keepSize=False): 
        '''
        remove lens distortion from given image
        ''' 
        image = imread(image)
        (h,w) = image.shape[:2]
        mapx, mapy = self.getUndistortRectifyMap(w,h)
   
        self.img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_TRANSPARENT
                            #borderValue=np.nan
                             )
        if not keepSize:
            xx,yy,ww,hh = self.roi
            self.img = self.img[yy : yy+hh, xx : xx+ww]
        return self.img


    def distortImage(self, image):
        '''
        opposite of 'correct'
        '''
        image = imread(image)
        (imgHeight, imgWidth) = image.shape[:2]
        mapx, mapy = self.getDistortRectifyMap(imgWidth, imgHeight)
        return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR, borderValue=(0,0,0))
                

    def getUndistortRectifyMap(self, imgWidth, imgHeight):

        if self.mapx is not None and self.mapx.shape == (imgHeight, imgWidth):
            return self.mapx, self.mapy
        
        cam = self.coeffs['cameraMatrix']
        d = self.coeffs['distortionCoeffs']
 
        (newCameraMatrix, self.roi) = cv2.getOptimalNewCameraMatrix(cam, 
                                    d, (imgWidth, imgHeight), 1, 
                                    (imgWidth, imgHeight))
        
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(cam, 
                                    d, None, newCameraMatrix, 
                                    (imgWidth, imgHeight), cv2.CV_32FC1)
        return self.mapx, self.mapy


    def getCameraParams(self):
        '''
        value positions based on http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#cv.InitUndistortRectifyMap
        '''
        c = self.coeffs['cameraMatrix']
        fx = c[0][0]
        fy = c[1][1]
        cx = c[0][2]
        cy = c[1][2]
        k1, k2, p1, p2, k3 = tuple(self.coeffs['distortionCoeffs'].tolist()[0])
        return fx, fy, cx, cy, k1, k2, k3, p1,  p2
 
 
    def setCameraParams(self, fx, fy, cx, cy, k1, k2, k3, p1, p2):
        c = self._coeffs['cameraMatrix']  = np.zeros(shape=(3,3))
        c[0,0] = fx
        c[1,1] = fy
        c[0,2] = cx
        c[1,2] = cy
        c[2,2] = 1
        self._coeffs['distortionCoeffs'] = np.array([[k1, k2, p1, p2, k3]])
 
    
    def getDistortRectifyMap(self, sizex, sizey):
        posy, posx = np.mgrid[0:sizey, 0:sizex].astype(np.float32)
        mapx, mapy = self.getUndistortRectifyMap(sizex, sizey)
        dx = posx-mapx
        dy = posy-mapy
        posx += dx
        posy += dy
        return posx, posy
 
 
    def getDeflection(self, width, height):
        mapx, mapy = self.getUndistortRectifyMap(width, height)
        ux = (1/np.abs(np.gradient(mapx)[1])) - 1
        uy = (1/np.abs(np.gradient(mapy)[0])) - 1
        ux[ux<0]=0
        uy[uy<0]=0
        return ux,uy


    def standardUncertainties(self):
        '''
        returns a list of standard uncertainties for the x and y component:
        (1x,2x), (1y, 2y), (intensity:None)
        1. px-size-changes(due to deflection)
        2. reprojection error
        '''
        height, width = self.coeffs['shape']
        ux,uy = self.getDeflection(width, height)
        r = self.coeffs['reprojectionError']#is RMSE of imgPoint-projectedPoints
        #transform to standard uncertainty
        #we assume rectangular distribution:
        ux = ux/(2 * 3**0.5)
        uy = uy/(2 * 3**0.5)
        return (ux,r), (uy,r), ()
 


if __name__ == '__main__':
    from fancytools.os.PathStr import PathStr
    import sys
    
    folder = PathStr(imgProcessor.__file__).dirname().join(
                        'media', 'lens_distortion')
    imgs = folder.all()
    
    l = LensDistortion()
    l.calibrate(board_size=(4,11), method='Asymmetric circles', 
                images=imgs, sensorSize_mm=(18,13.5) )
    print l.getCoeffStr()
    
    img = l.drawChessboard()
    
    if 'no_window' not in sys.argv:
        cv2.imshow('chessboard', img)
        cv2.waitKey(0)
