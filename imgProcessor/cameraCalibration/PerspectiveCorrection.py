import numpy as np
import cv2
from transforms3d import euler

from fancytools.math.Point3D import Point3D

from imgProcessor.imgIO import imread
from imgProcessor.sortCorners import  sortCorners
from imgProcessor.genericCameraMatrix import genericCameraMatrix
from imgProcessor.equations.vignetting import tiltFactor
from imgProcessor.equations.defocusThroughDepth import defocusThroughDepth
from imgProcessor.imgPointToWorldCoord import imgPointToWorldCoord
from imgProcessor.PatternRecognition import PatternRecognition
from imgProcessor.calcAspectRatioFromCorners import calcAspectRatioFromCorners



class PerspectiveCorrection(object):
    
    def __init__(self,
                 img_shape,
                 obj_height_mm=None,
                 obj_width_mm=None,
                 cameraMatrix=None,
                 distCoeffs=0,
                 do_correctIntensity=False,
                 new_size=None,
                 in_plane=False,
                 cv2_opts={} ):
        '''
        correction(warp+intensity factor) + uncertainty due to perspective distortion
        
        new_size = (sizey, sizex)
            if either sizey or sizex is None the resulting size will set 
            using an (assumed) aspect ratio 
        
        in_plane=True --> object has to tilt, only rotation and translation
        
        this class saves all parameter maps in self.maps
        !!!
        given images need to be already free from lens distortion
        and distCoeffs should be 0
        !!!
        '''
        self.opts = {'obj_height_mm':obj_height_mm,
                     'obj_width_mm':obj_width_mm,
                     'distCoeffs':distCoeffs,
                     'do_correctIntensity':do_correctIntensity,
                     'new_size':new_size,
                     'in_plane':in_plane,
                     'cv2_opts':cv2_opts}
        if cameraMatrix is None:
            cameraMatrix = genericCameraMatrix(img_shape)
        self.opts['cameraMatrix'] = cameraMatrix


    def setReference(self, ref):
        '''
        ref  ... either quad, grid, homography or reference image
        
        quad --> list of four image points(x,y) marking the edges of the quad
               to correct
        homography --> h. matrix to correct perspective distortion
        referenceImage --> image of same object without perspective distortion
        '''
        self.maps = {}
        self.quad=None 
        self._obj_points = None
        self._camera_position = None
        self._homography = None
        self._homography_is_fixed = True
        self.tvec, self.rvec = None, None
  
        #evaluate input:
        if isinstance(ref, np.ndarray) and ref.shape == (3,3):
            #REF IS HOMOGRAPHY
            self._homography = ref
            #REF IS QUAD
        elif len(ref)==4:
            self.quad = sortCorners(ref)
            #REF IS IMAGE
        else: 
            self.pattern = PatternRecognition(imread(ref))
            self._homography_is_fixed = False

    
    @property
    def homography(self):
        if self._homography is None:
            if self.quad is not None:
                #GET HOMOGRAPHIE FROM QUAD
                fixedX,fixedY = None,None
                try:
                    #is new size is given
                    sx, sy = self.opts['new_size']
                    if sx is None and sy is not None:
                        fixedY = sy
                        raise TypeError()
                    elif sx is not None and sy is None:
                        fixedX = sx
                        raise TypeError()
                except TypeError:
                    try:
                        #estimate size
                        w = self.opts['obj_width_mm']
                        h = self.opts['obj_height_mm']
                        aspectRatio = float(w)/h
                    except TypeError:
                        aspectRatio = calcAspectRatioFromCorners(self.quad, 
                                                                self.opts['in_plane'])
                        print 'aspect ratio assumed to be %s' %aspectRatio
                    #new image border keeping aspect ratio
                    if fixedX or fixedY:
                        if fixedX:
                            sx = fixedX
                            sy = sx/aspectRatio
                        else:
                            sy = fixedY
                            sx = sy*aspectRatio
                    else:                                        
                        sx,sy = self._calcQuadSize(self.quad, aspectRatio)
                            
                self._newBorders = (int(round(sx)),int(round(sy)))
                #image edges:
                objP = np.array([
                            [0, 0],
                            [sx, 0],
                            [sx, sy],
                            [0, sy],
                            ],dtype=np.float32)
                self._homography = cv2.getPerspectiveTransform(
                                        self.quad.astype(np.float32), objP)
            else:
                #GET HOMOGRAPHY USING PATTERN RECOGNITION
                self._Hinv = h = self.pattern.findHomography(self.img)[0]
                self._homography = self.pattern.invertHomography(h)
                s = self.img.shape
                self._newBorders = (s[1], s[0])
        
        return self._homography


    def distort(self,img, rotX=0, rotY=0, quad=None):
        '''
        Apply perspective distortion ion self.img
        angles are in DEG and need to be positive to fit into image
        
        '''
        self.img = imread(img)
        #fit old image to self.quad:
        corr = self.correct(self.img)   
        s = self.img.shape
        if quad is None:
            wquad = (self.quad - self.quad.mean(axis=0)).astype(float)
            
            win_width = s[1]
            win_height = s[0]
            #project quad:
            for n, q in enumerate(wquad):
                p = Point3D(q[0],q[1],0).rotateX(-rotX).rotateY(-rotY)
                p = p.project(win_width, win_height, s[1], s[1])
                wquad[n] = (p.x,p.y)
            wquad= sortCorners(wquad)
            #scale result so that longest side of quad and wquad are equal
            w = wquad[:,0].max() - wquad[:,0].min()
            h = wquad[:,1].max() - wquad[:,1].min()
            scale = min(s[1]/w,s[0]/h)
            #scale:
            wquad = (wquad*scale).astype(int)
        else:
            wquad = sortCorners(quad)
        wquad -= wquad.min(axis=0)
  
        lx = corr.shape[1]
        ly = corr.shape[0]

        objP = np.array([
                    [0, 0],
                    [lx, 0],
                    [lx, ly],
                    [0, ly],
                    ],dtype=np.float32)

        homography = cv2.getPerspectiveTransform(wquad.astype(np.float32), objP)
        #distort corr:
        w = wquad[:,0].max() - wquad[:,0].min()
        h = wquad[:,1].max() - wquad[:,1].min() 
        #(int(w),int(h))
        dist = cv2.warpPerspective(corr, homography, (int(w),int(h)), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

        #move middle of dist to middle of the old quad: 
        bg = np.zeros(shape=s)
        rmn = (bg.shape[0]/2, bg.shape[1]/2)
        
        ss = dist.shape
        mn = (ss[0]/2,ss[1]/2) #wquad.mean(axis=0)
        ref = (int(rmn[0]-mn[0]), int(rmn[1]-mn[1]))

        bg[ref[0]:ss[0]+ref[0], ref[1]:ss[1]+ref[1]] = dist
        
        #finally move quad into right position:
        self.quad = wquad
        self.quad += (ref[1],ref[0])
        self.img = bg
        self._homography = None
        self._poseFromQuad()
        
        if self.opts['do_correctIntensity']:
            self.img *= self._getTiltFactor(self.img)

        return self.img


    def _getTiltFactor(self, img):
        #CALCULATE VIGNETTING OF WARPED OBJECT:
        _,r = self.pose()
        eulerAngles = euler.mat2euler(cv2.Rodrigues(r)[0], axes='rzxy')
        
        tilt = eulerAngles[1]
        rot = eulerAngles[0]
        f = self.opts['cameraMatrix'][0,0]
        s = img.shape
        self.maps['tilt_factor'] = tf = np.fromfunction(lambda x,y: tiltFactor((x, y), 
                                              f=f, tilt=tilt, rot=rot), s[:2])
        #if img is color:
        if tf.shape != s:
            tf = np.repeat(tf, s[-1]).reshape(s)
        return tf


    def correctGrid(self, img, grid):
        '''
        grid -> array of polylines=((p0x,p0y),(p1x,p1y),,,)
        '''

        self.img = imread(img)
        h = self.homography#TODO: cleanup only needed to get newBorder attr.

        if self.opts['do_correctIntensity']:
            self.img = self.img / self._getTiltFactor(self.img)

        snew = self._newBorders
        warped = np.empty(snew[::-1], dtype=self.img.dtype)
        s0,s1 = grid.shape[:2]
        nx,ny = s0-1,s1-1
        sy,sx = snew[0]/nx,snew[1]/ny
        
        objP = np.array([[0, 0 ],
                         [sx,0 ],
                         [sx,sy],
                         [0, sy] ],dtype=np.float32)
        
        for ix in xrange(nx):
            for iy in xrange(ny):
                quad = grid[ix:ix+2,iy:iy+2].reshape(4,2)[np.array([0,2,3,1])]
                hcell = cv2.getPerspectiveTransform(
                                quad.astype(np.float32), objP)

                cv2.warpPerspective(self.img, hcell, (sx,sy),
                                    warped[iy*sy : (iy+1)*sy,
                                           ix*sx : (ix+1)*sx],
                                    flags=cv2.INTER_LANCZOS4,
                                    **self.opts['cv2_opts'])
        return warped    


    def correct(self, img):
        '''
        ...from perspective distortion: 
         --> perspective transformation
         --> apply tilt factor (view factor) correction 
        '''
        self.img = imread(img)

        if not self._homography_is_fixed:
            self._homography = None
        h = self.homography
        if self.opts['do_correctIntensity']:
            self.img = self.img / self._getTiltFactor(self.img)
        warped = cv2.warpPerspective(self.img, 
                                     h, 
                                     self._newBorders,
                                     flags=cv2.INTER_LANCZOS4,
                                     **self.opts['cv2_opts'])
        return warped


    def correctPoints(self, pts):
        if not self._homography_is_fixed:
            self._homography = None
        
        h = self.homography
#         #normalize
#         h /= h[2,2]
#         #invert homography
#         h = np.linalg.inv(h)

        if pts.ndim == 2:
            pts = pts.reshape(1,*pts.shape) 
        return cv2.perspectiveTransform(pts.astype(np.float32), h)


    @property
    def camera_position(self):
        '''
        returns camera position in world coordinates using self.rvec and self.tvec
        from http://stackoverflow.com/questions/14515200/python-opencv-solvepnp-yields-wrong-translation-vector
        '''
        t,r = self.pose()
        if self._camera_position is None:
            self._camera_position = -np.matrix(cv2.Rodrigues(r)[0]).T * np.matrix(t)
        return self._camera_position


    def standardUncertainties(self, focal_Length_mm, f_number, 
                    focusAtYX=None,
                    sigma_best_focus=0,
                    quad_pos_err=0,
                    shape=None,
                    uncertainties=( (),(),() ) ):
        '''
        focusAtXY - image position with is in focus
            if not set it is assumed that the image middle is in focus
        sigma_best_focus - standard deviation of the PSF
                             within the best focus (default blur)
        uncertainties - contibutors for standard uncertainty
                        these need to be perspective transformed to fit the new 
                        image shape
        '''
        #TODO: consider quad_pos_error
        ############################## (also influences intensity corr map)
        
        cam  = self.opts['cameraMatrix']
        if shape is None:
            s = self.img.shape
        else:
            s = shape

        # 1. DEFOCUS DUE TO DEPTH OF FIELD
        ################################## 
        t,r = self.pose()
        worldCoord = np.fromfunction(lambda x,y: imgPointToWorldCoord((y,x), r, t, cam), s)
        depthMap = np.linalg.norm(worldCoord-self.camera_position, axis=0).reshape(s)
        del worldCoord

        if focusAtYX is None:
            #assume image middle is in-focus:
            focusAtYX = (s[0]/2,s[1]/2)
        infocusDepth = depthMap[focusAtYX]
        depthOfField_blur = defocusThroughDepth(depthMap, infocusDepth, focal_Length_mm, f_number, k=2.335)

        #2. INCREAASED PIXEL SIZE DUE TO INTERPOLATION BETWEEN 
        #   PIXELS MOVED APARD
        ######################################################
        #index maps:
        py, px = np.mgrid[0:s[0], 0:s[1]]
        #warped index maps:
        wx = cv2.warpPerspective(np.asfarray(px), self.homography, self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4  )
        wy = cv2.warpPerspective(np.asfarray(py), self.homography, self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4  )

        pxSizeFactorX= (1/np.abs(np.gradient(wx)[1]))
        pxSizeFactorY= (1/np.abs(np.gradient(wy)[0]))

        self.maps['depthMap'] = depthMap

        #AREA RATIO AFTER/BEFORE:
            #AREA OF QUADRILATERAL:
        q = self.quad
        quad_size = 0.5* abs( (q[2,0]-q[0,0])*(q[3,1]-q[1,1]) + 
                               (q[3,0]-q[1,0])*(q[0,1]-q[2,1]))
        sx,sy = self._newBorders
        self.areaRatio = (sx*sy)/quad_size

        #WARP ALL FIELD TO NEW PERSPECTIVE AND MULTIPLY WITH PXSIZE FACTOR:  
        f = (pxSizeFactorX**2+pxSizeFactorY**2)**0.5

        self.maps['depthOfField_blur'] = depthOfField_blur = cv2.warpPerspective(
                                 depthOfField_blur, self.homography, self._newBorders,
                                 borderValue=np.nan,
                                 )*f

        #perspective transform given uncertainties:
        warpedU = []
        for u in uncertainties:
            warpedU.append([])
            for i in u:
                #print i, type(i), isinstance(i, np.ndarray)
                if isinstance(i, np.ndarray) and i.size > 1:
                    i = cv2.warpPerspective(i, self.homography, self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4)*f
                                 
                else:
                #multiply with area ratio: after/before perspective warp
                    i*=self.areaRatio
                      
                warpedU[-1].append(i)

        delectionUncertX = (pxSizeFactorX-1)/(2*3**0.5)
        delectionUncertY = (pxSizeFactorY-1)/(2*3**0.5)

        warpedU[0].extend((delectionUncertX, depthOfField_blur))
        warpedU[1].extend((delectionUncertY, depthOfField_blur))

        return tuple(warpedU)

    
    def pose(self):
        if self.tvec is None:
            if self.quad is not None:
                self._poseFromQuad()
            else:
                self._poseFromHomography()
        return self.tvec, self.rvec


    def _poseFromHomography(self):
        sy,sx = self.img.shape[:2]
        #image edges:
        objP = np.array([[
                    [0, 0],
                    [sx, 0],
                    [sx, sy],
                    [0, sy],
                    ]],dtype=np.float32)
        quad = cv2.perspectiveTransform(objP, self._Hinv)
        self._poseFromQuad(quad)
        

    def _poseFromQuad(self, quad=None):
        '''
        estimate the pose of the object plane using quad
            setting:
        self.rvec -> rotation vector
        self.tvec -> translation vector
        '''
        if quad is  None:
            quad = self.quad
        # http://answers.opencv.org/question/1073/what-format-does-cv2solvepnp-use-for-points-in/
        # Find the rotation and translation vectors.        
        retval, self.rvec, self.tvec = cv2.solvePnP(
                    self.obj_points, 
                    quad.astype(np.float32), 
                    self.opts['cameraMatrix'], 
                    self.opts['distCoeffs'],
                    #flags=cv2.CV_ITERATIVE 
                    )
        if retval is None:
            print("Couln't estimate pose")


    @property
    def obj_points(self):
        if self._obj_points is None:
            h = self.opts['obj_height_mm']
            w = self.opts['obj_width_mm']
            if w is None or h is None:
                w,h = 100,100

            self._obj_points = np.array([
                        [0, 0, 0 ],
                        [w, 0, 0],
                        [w, h, 0],
                        [0, h, 0],
                        ],dtype=np.float32)
        return self._obj_points


    def drawQuad(self,img=None, quad=None, thickness=30):
        '''
        Draw the quad into given img 
        '''
        if img is None:
            img = self.img
        if quad is None:
            quad = self.quad
        q = quad
        c = int(img.max())
        cv2.line(img, tuple(q[0]), tuple(q[1]), c, thickness)
        cv2.line(img, tuple(q[1]), tuple(q[2]), c, thickness)
        cv2.line(img, tuple(q[2]), tuple(q[3]), c, thickness)
        cv2.line(img, tuple(q[3]), tuple(q[0]), c, thickness)
        return img


    def draw3dCoordAxis(self,img=None, thickness=8):
        '''
        draw the 3d coordinate axes into given image
        if image == False:
            create an empty image
        '''
        if img is None:
            img = self.img
        elif img is False:
            img = np.zeros(shape=self.img.shape, dtype=self.img.dtype)
        else:
            img = imread(img)
        # project 3D points to image plane:
        w,h = self.opts['obj_width_mm'], self.opts['obj_height_mm']
        axis = np.float32([[0.5*w,0.5*h,0],
                           [ w, 0.5*h, 0], 
                           [0.5*w, h, 0], 
                           [0.5*w,0.5*h, -0.5*w]])
        t,r = self.pose()
        imgpts = cv2.projectPoints(axis, r, t, 
                                   self.opts['cameraMatrix'], 
                                   self.opts['distCoeffs'])[0]
    
        mx = img.max()
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,0,mx), thickness)
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,mx,0), thickness)
        cv2.line(img, origin, tuple(imgpts[3].ravel()), (mx,0,0), thickness*2)
        return img

    
    @staticmethod
    def _calcQuadSize(corners, aspectRatio):
        '''
        return the size of a rectangle in perspective distortion in [px]
        DEBUG: PUT THAT BACK IN??::
            if aspectRatio is not given is will be determined
        '''           
        if aspectRatio > 1: #x is bigger -> reduce y
            x_length = PerspectiveCorrection._quadXLength(corners)
            y = x_length / aspectRatio
            return x_length, y
        else: # y is bigger -> reduce x
            y_length = PerspectiveCorrection._quadYLength(corners)          
            x = y_length*aspectRatio
            return x, y_length


    @staticmethod
    def _quadYLength(corners):
        ll = PerspectiveCorrection._linelength
        l0 = (corners[1], corners[2])
        l1 = (corners[0], corners[3])
        return max(ll(l0), ll(l1)) 


    @staticmethod
    def _quadXLength(corners):
        ll = PerspectiveCorrection._linelength
        l0 = (corners[0], corners[1])
        l1 = (corners[2], corners[3])
        return max(ll(l0), ll(l1))


    @staticmethod
    def _linelength(line):
        p0,p1 = line
        x0,y0 = p0
        x1,y1 = p1
        dx = x1-x0
        dy = y1-y0
        return (dx**2+dy**2)**0.5




if __name__ == '__main__':
    #in this example we will rotate a given image in perspective
    #and then correct the perspective with given corners or a reference
    #(the undistorted original) image
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    from imgProcessor.transformations import toUIntArray

    #1. LOAD TEST IMAGE
    path = PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence', 'EL_module_orig.PNG')
    img = imread(path)
    
    #DEFINE OBJECT CORNERS:
    sy,sx = img.shape[:2]
                    #x,y
    obj_corners = [(8,  2),
                   (198,6),
                   (198,410),
                   (9,  411)]
    #LOAD CLASS:
    pc = PerspectiveCorrection(img.shape,
                               do_correctIntensity=True,
                               obj_height_mm=sy,
                               obj_width_mm=sx)
    pc.setReference(obj_corners)
    img2 = img.copy()
    pc.drawQuad(img2, thickness=2)
 
    #2. DISTORT THE IMAGE:
    dist = pc.distort(img2, rotX=10, rotY=20)
    dist2 = dist.copy()
    pc.draw3dCoordAxis(dist2, thickness=2)
    pc.drawQuad(dist2, thickness=2)
    #3a. CORRECT WITH QUAD:
    corr = pc.correct(dist)
 
    #3b. CORRECT WITH WITHEN REFERENCE IMAGE
#     pc.img = dist
    pc.setReference(img)
    corr2 = pc.correct(dist)

    #DISLAY
    cv2.imshow("1. original", img2)
    cv2.imshow("2. distorted", toUIntArray(dist2, dtype=np.uint8))
    cv2.imshow("3a. corr. with quad", toUIntArray(corr, dtype=np.uint8))
    cv2.imshow("3b. corr. with ref. image", toUIntArray(corr2, dtype=np.uint8))
    cv2.waitKey(0)

