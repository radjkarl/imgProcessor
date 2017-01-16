from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from transforms3d.euler import mat2euler, euler2mat
from inspect import getmembers, isfunction

from fancytools.math.Point3D import Point3D

from imgProcessor.imgIO import imread
from imgProcessor.utils.sortCorners import  sortCorners
from imgProcessor.utils.genericCameraMatrix import genericCameraMatrix
from imgProcessor.equations.vignetting import tiltFactor
from imgProcessor.equations.defocusThroughDepth import defocusThroughDepth
from imgProcessor.utils.imgPointToWorldCoord import imgPointToWorldCoord
from imgProcessor.features.PatternRecognition import PatternRecognition
from imgProcessor.utils.calcAspectRatioFromCorners import calcAspectRatioFromCorners
from imgProcessor.physics import emissivity_vs_angle

    
def rvec2euler(rvec):
    return mat2euler(cv2.Rodrigues(rvec)[0], axes='rzxy')

def euler2rvec(a0,a1,a2):
    return cv2.Rodrigues(euler2mat(a0,a1,a2, axes='rzxy'))[0]
    
class PerspectiveCorrection(object):
    
    def __init__(self,
                 img_shape,
                 obj_height_mm=None,
                 obj_width_mm=None,
                 cameraMatrix=None,
                 distCoeffs=np.zeros((5,1)),
                 do_correctIntensity=False,
                 px_per_phys_unit=None,
                 new_size=(None,None),
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
        #TODO: remove camera matrix and dist coeffs completely - dont needed
        #since img needs to be lens corrected anyway
        
        #TODO: insert aspect ratio and remove obj_width, height
        self.opts = {'obj_height_mm':obj_height_mm,
                     'obj_width_mm':obj_width_mm,
                     'distCoeffs':distCoeffs.astype(np.float32),
                     'do_correctIntensity':do_correctIntensity,
                     'new_size':new_size,
                     'in_plane':in_plane,
                     'cv2_opts':cv2_opts,
                     'shape':img_shape[:2]}
        if cameraMatrix is None:
            cameraMatrix = genericCameraMatrix(img_shape)
        
        self.opts['cameraMatrix'] = cameraMatrix.astype(np.float32)
        self.refQuad = None
        self._obj_points = None
        self.px_per_phys_unit = px_per_phys_unit

        self._newBorders = self.opts['new_size']


    def setReferenceQuad(self, refQuad):
        '''
        TODO
        '''
        self.refQuad=sortCorners(refQuad)
        


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
#         self.refQuad = None
        self._camera_position = None
        self._homography = None
        self._homography_is_fixed = True
#         self.tvec, self.rvec = None, None
        self._pose = None
        
        #evaluate input:
        if isinstance(ref, np.ndarray) and ref.shape == (3,3):
            #REF IS HOMOGRAPHY
            self._homography = ref
            #REF IS QUAD
        elif len(ref)==4:
            self.quad = sortCorners(ref)
            
            #TODO: cleanup # only need to call once - here
            o = self.obj_points #no property any more
            
            #REF IS IMAGE
        else: 
            ref = imread(ref)
            self._refshape = ref.shape[:2]
            self.pattern = PatternRecognition(ref)
            self._homography_is_fixed = False

    
    @property
    def homography(self):
        if self._homography is None:
            if self.quad is not None:
                src = self.quad.astype(np.float32)

                if self.refQuad is not None:
                    dst = self.refQuad.astype(np.float32)
                else:
#                     dst = np.array(self.obj_points[:,:2],np.float32)
                    sx,sy = self._newBorders
                    dst = np.float32([
                        [0,  0],
                        [sx, 0],
                        [sx, sy],
                        [0,  sy]])
                    
                self._homography = cv2.getPerspectiveTransform(src, dst)
    
            else:
                #GET HOMOGRAPHY FROM REFERENCE IMAGE USING PATTERN RECOGNITION
                self._Hinv = h = self.pattern.findHomography(self.img)[0]
                self._homography = self.pattern.invertHomography(h)
                sx, sy = self.opts['new_size']
                ssy,ssx  = self._refshape
                if sx is None:
                    sx = ssx
                if sy is None:
                    sy = ssy
                self._newBorders = (sx,sy)
        
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
        dist = cv2.warpPerspective(corr, homography, (int(w),int(h)), 
                                   flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

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
            self.img *= self._getTiltFactor(self.img.shape)

        return self.img

    def objectOrientation(self):
        tvec,r = self.pose()
        eulerAngles = mat2euler(cv2.Rodrigues(r)[0], axes='rzxy')
        
        #my measurement shows that *2 brings right results
        #TODO: WHY???
        tilt = eulerAngles[1]
        rot = eulerAngles[0]
        dist = tvec[2,0]#only take depth component np.linalg.norm(tvec)
        return dist, tilt, rot


    #TODO: remove!
    def _getTiltFactor(self, shape, orientation=None):
        '''
        optional:
        orientation = ( distance to image plane in px,
                        object tilt [radians],
                        object rotation [radians] )
        otherwise calculated from homography
        '''
        #CALCULATE VIGNETTING OF WARPED OBJECT:
#         f = self.opts['cameraMatrix'][0,0]
        try: dist, tilt, rot = orientation
        except TypeError:
            dist, tilt, rot = self.objectOrientation()
            rot -= np.pi/2# because of different references
        if self.quad is not None:
            cent = self.quad[:,1].mean(), self.quad[:,0].mean()
        else:
            cent = None
        self.maps['tilt_factor'] = tf = np.fromfunction(lambda x,y: tiltFactor((x, y), 
                                              f=dist, tilt=tilt, rot=rot, center=cent), shape[:2])
        #if img is color:
        if tf.shape != shape:
            tf = np.repeat(tf, shape[-1]).reshape(shape)
        return tf


    def correctGrid(self, img, grid):
        '''
        grid -> array of polylines=((p0x,p0y),(p1x,p1y),,,)
        '''

        self.img = imread(img)
        h = self.homography#TODO: cleanup only needed to get newBorder attr.

        #TODO: implement new titl factor
        if self.opts['do_correctIntensity']:
            self.img = self.img / self._getTiltFactor(self.img.shape)

        s0,s1 = grid.shape[:2]
        n0,n1 = s0-1,s1-1
        
        snew = self._newBorders

        sx,sy = snew[0]//n0, snew[1]//n1
        
#         out = np.empty(snew[::-1], dtype=self.img.dtype)        
        out = np.empty((sy*n1,sx*n0), dtype=self.img.dtype)        
        
#         sx,sy = snew[0]//n0, snew[1]//n1
        
        objP = np.array([[0, 0 ],
                         [sx,0 ],
                         [sx,sy],
                         [0, sy] ],dtype=np.float32)
        
        # warp every cell in grid:
        for ix in range(n0):
            for iy in range(n1):
                quad = grid[ix:ix+2,iy:iy+2].reshape(4,2)[np.array([0,2,3,1])]
                hcell = cv2.getPerspectiveTransform(
                                quad.astype(np.float32), objP)

                cv2.warpPerspective(self.img, hcell, (sx,sy),
                                    out[iy*sy : (iy+1)*sy,
                                           ix*sx : (ix+1)*sx],
                                    flags=cv2.INTER_LANCZOS4,
                                    **self.opts['cv2_opts'])

        return out    


    def uncorrect(self, img):
        img = imread(img)
        s = img.shape[:2]
        return cv2.warpPerspective(img, self.homography, s[::-1], 
                                   flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)


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
            #TODO: replace with new tilt factor
            self.img = self.img / self._getTiltFactor(self.img.shape)
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
    def camera_position(self, pose=None):
        '''
        returns camera position in world coordinates using self.rvec and self.tvec
        from http://stackoverflow.com/questions/14515200/python-opencv-solvepnp-yields-wrong-translation-vector
        '''
        #if self._camera_position is None:
        if pose is None:
            pose = self.pose()
        t,r = pose
        return -np.matrix(cv2.Rodrigues(r)[0]).T * np.matrix(t)
#         self._camera_position = -np.matrix(cv2.Rodrigues(r)[0]).T * np.matrix(t)
#         return self._camera_position


    def planeSfN(self, rvec):
        #get z:
            #1 undistort plane:
        rot = cv2.Rodrigues(rvec)[0]
        aa = np.array([0.,0.,1.])
        return aa.dot(rot)
        
#         #obj points brought into perspective:
#         oo = self.obj_points.dot(rot)
#         q = self.quad
# 
#         #build 2 position vectors:
#         p0 = np.empty(shape=3)
#         p0[:2] = q[1]-q[0]
#         p0[2] = oo[1,2]-oo[0,1]
# 
#         p1 = np.empty(shape=3)
#         p1[:2] = q[0]-q[3]
#         p1[2] = oo[0,2]-oo[3,1]
#         #SURFACE NORMAL OF PLANE:
#         n = np.cross(p0,p1)
#         return n/np.linalg.norm(n)#normalize


    def depthMap(self, midpointdepth=0, pose=None):
        shape = self.opts['shape']
        if pose is None:
            pose = self.pose()
        t,r = pose
        
        n = self.planeSfN(r)
        #z component from plane-equation solved for z:
        zpart = np.fromfunction(lambda y,x:(-n[0]*x
                                            -n[1]*y)/(
                                            -n[2]), shape)
        
        ox,oy = self.objCenter()
        v = zpart[int(oy),int(ox) ]
        zpart+=midpointdepth-v

        return zpart
    
    
    def cam2PlaneVectorField(self, midpointdepth=0, **kwargs):
        t,r = self.pose()
        shape = self.opts['shape']

        cam  = self.opts['cameraMatrix']
        #move reference point from top left quad corner to
        #optical center:
#         q0 = self.quad[0]
        q0 = self.objCenter()
#         dx,dy = cam[0,2]-q0[0], cam[1,2]-q0[1]
        dx,dy = shape[1]//2-q0[0], shape[0]//2-q0[1]
        
        #x,y component of undist plane:
        rot0 = np.array([0,0,0], dtype=float)
        worldCoord = np.fromfunction(lambda x,y: 
                imgPointToWorldCoord((y-dy,x-dx), rot0, t, cam 
                       ), shape).reshape(3, *shape)
        #z component from plane-equation solved for z:
        n = self.planeSfN(r)
        x,y = worldCoord[:2]
        zpart = (-n[0]*x-n[1]*y)/(-n[2])
        ox,oy = self.objCenter()
        v = zpart[int(oy),int(ox) ]
        zpart+=midpointdepth-v
        worldCoord[2]=zpart
        return worldCoord



    #BEFORE REMOVING THINGS: MAKE EXTRA FN
    def viewAngle(self, **kwargs):
        '''
        calculate view factor between one small and one finite surface
        vf =1/pi * integral(cos(beta1)*cos(beta2)/s**2) * dA
        accorduing to VDI heatatlas 2010 p961
        '''
        
        v0 = self.cam2PlaneVectorField(**kwargs)
        
        #obj cannot be behind camera
        v0[2][v0[2]<0]=np.nan
        
        s = np.linalg.norm(v0,axis=0)#distance camera<->plane

        t,r = self.pose()
        n = self.planeSfN(r)
        
        norm = np.linalg.norm
        s0,s1 = s.shape
        
        def vectorAngle(vec1,vec2):
            a = np.arccos(
                        np.einsum('ijk,ijk->jk', vec1,vec2)
                        /( norm(vec1,axis=0) * norm(vec2,axis=0) ) )
            #take smaller of both possible angles:
            ab = np.abs(np.pi-a)
            with np.errstate(invalid='ignore'):
                i = a>ab
            a[i] = ab[i]
            return a
        
        def vectorToField(vec):
            out = np.empty(shape=(3,s0,s1))
            out[0] = vec[0]
            out[1] = vec[1]
            out[2] = vec[2]
            return out

#         dd = kwargs['midpointdepth']
        #unify deph from focal point:
#         vv0 = np.empty_like(v0)
#         vv0[:2]=v0[:2]
#         vv0[2]=dd

#         beta1 = vectorAngle(vv0, vectorToField([0.,0.,1.]) )

        #because of different x,y orientation:
        n[2]*=-1
        beta2 = vectorAngle(v0, vectorToField(n) )

#         beta2 -= np.pi/2
#         beta2 = np.abs(beta2)

        #length of one pixel in focal plane
#         size_px_x, size_px_y = v0[:2,1, 1]-v0[:2,0,0]
#         print (v0[:2,1, 1], v0[:2,0,0], size_px_x, size_px_y )
#         size_px_x, size_px_y = 3.7,3.7
     
        #area on plane of one image pixel
        #using theorem on intersecting lines:
#         A = size_px_x * size_px_y * (s/dd)**2


#         dy = v0[1,0,0]-v0[1,-1,0]
#         dz = v0[2,0,0]-v0[2,-1,0]
#         dx = v0[0,0,0]-v0[0,-1,0]
      
# 
#         beta2[~self.foreground()]=np.nan

#         import pylab as plt
#         from mpl_toolkits.mplot3d import Axes3D
# # # #         
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')        
#         ax.plot_surface(v0[0,::10,::10], v0[1,::10,::10], v0[2,::10,::10])
#         plt.show()
# #         
#         plt.imshow(np.degrees(beta2))
#         plt.colorbar()
#         plt.figure(2)
#         plt.imshow(np.degrees(beta1))
#         plt.colorbar()
# # #   
#         plt.figure(3)
#         plt.imshow(v0[0])
#         plt.colorbar()
# # # # #   
#         plt.figure(4)
#         plt.imshow(v0[1])
#         plt.colorbar()
# # # #    
#         plt.show()
 
#         vf = (dA*np.cos(beta1)*np.cos(beta2)) / s**2
        #beta2[:]=beta2.mean()
#         vf = (A*np.cos(beta2)) / s**2

#         import pylab as plt
#         plt.figure(4)
#         plt.imshow(vf)
#         plt.colorbar()
# # #    
#         plt.show()

        return beta2


#     @staticmethod
#     def _updateTvec(tvec, obj_center=None, distance=None):
#         '''
#         obj_center (x,y) - center of plane [px]
#         distance [px]
#         '''
#         if distance is not None:
#             tvec[2,0]=distance
#         if obj_center is not None:
#             tvec[0,0]=obj_center[0]
#             tvec[1,0]=obj_center[1]
#         return tvec
    
#     @staticmethod
#     def _updateRVec(rvec, rotation=None, tilt=None):
#         '''
#         in DEG
#         '''
#         if rotation is None and tilt is None:
#             return rvec
#         r, t, p = rvec2euler(rvec)
#         if rotation is not None:
#             r = np.radians(rotation)
#         if tilt is not None:
#             t =  np.radians(tilt)  
#         return euler2rvec(r,t,p)

# 
#     def corrDepthMap(self, midpointdepth=0, pose=None):
# 
#         if pose is None:
#             pose = self.pose()
#         tvec,rvec = pose
# 
# 
# #         tvec = self._updateTvec(tvec, obj_center, distance)
# #         rvec = self._updateRVec(rvec, rotation, tilt)
# 
#         #with current/given obj. rotation:
#         d = self.depthMap( pose=(tvec,rvec))
# #         return d+midpointdepth
# #         return d
#         #without object rotation:
#         
#         ##################
#         #middle point has to be img middle when zero rot
#         ####################
#         
# #         diag = np.linalg.norm(self.quad[0,0]-self.objCenter())
# #         f = diag / np.linalg.norm(self.obj_points[0,:2])
#         
#         #no-tilt rotation vector:
#         rvec0 = self._updateRVec(rvec, tilt=0)
# #         r,_,p = rvec2euler(rvec)
# #         rvec0 = euler2rvec(r,0,p)
# 
#         d2 = self.depthMap( (tvec,rvec0))
#         dd = d-d2
# #         dd = d
# 
# #         if self.px_per_phys_unit is None:
# #             objwidth = self.obj_points[1,0]*2
# #             pxwidth = np.linalg.norm(self.quad[1]-self.quad[0])
# #             self.px_per_phys_unit = pxwidth / objwidth
#         
#         #add depth at object mid point:
#         x,y = self.objCenter()
#         v = dd[int(y),int(x) ]
#         dd+=midpointdepth-v
#         
# #         if excludeBG:
# #             bg = np.ones_like(dd, dtype=np.uint8)
# #             cv2.fillConvexPoly(bg, self.quad, 0)
# #             bg = bg.astype(bool)
# #             import pylab as plt
# #             plt.imshow(bg)
# #             plt.show()
#             
#         return dd

    def foreground(self, quad=None):
        fg = np.zeros(shape=self._newBorders[::-1], dtype=np.uint8)
        if quad is None:
            quad = self.quad
        else:
            quad = quad.astype(np.int32)
        cv2.fillConvexPoly(fg, quad, 1)
        return fg.astype(bool)


    def tiltFactor(self, midpointdepth=0, material='EL_Si_module', 
                   #pose=None, 
                   #remove:
                   tilt=None
                   ):
        '''
        get tilt factor from inverse distance law
        https://en.wikipedia.org/wiki/Inverse-square_law
        '''
       # if pose is None:
       #     pose = self.pose()
#         _, rvec = self.pose()
#         d = self.depthMap(midpointdepth, pose)
#         cx,cy = self.objCenter()
#         dref = d[int(cy), int(cx)]
#         intensities = dref**2/d**2
        
        #TODO: can also be only def. with FOV, rot, tilt
        beta2 = self.viewAngle(midpointdepth=midpointdepth)
        #print( 'angle:' , np.degrees(beta2[self.foreground()].mean()))
#         cx,cy = self.objCenter()
#         dref = d_noZ[int(cy), int(cx)]        
#         i2=  dref**2/d_noZ**2

        #ensure that average tilt factor == 1:
        #vf /= vf[self.foreground()].mean()


        try:
            angles, vals = getattr(emissivity_vs_angle, material)()
        except AttributeError:
            raise AttributeError("material[%s] is not in list of know materials: %s" %(
                 material,[o[0] for o in getmembers(emissivity_vs_angle) 
                           if isfunction(o[1])]))

        avg_angle = beta2[self.foreground()].mean()
        print( 'angle:' , np.degrees(avg_angle))
        
        #use averaged angle instead of beta2 to not overemphasize correction
        normEmissivity = np.clip(
                            InterpolatedUnivariateSpline(
                                    np.radians(angles), vals)(beta2),0,1)
        


#         import pylab as plt
# # 
# # #         d[~self.foreground()]=np.nan
# #         vf[~self.foreground()]=np.nan
# # # 
#         plt.figure(3)
#         plt.imshow(vf)
#         plt.colorbar()
# # 
# #         print(dref)
#         plt.figure(4)
#         plt.imshow(beta2)
#         plt.colorbar()
# # # 
#         plt.show()

#         return vf*normEmissivity


#         c0,c1 = 2000,2000
#         refD = d[c0,c1]
#         intensities = dref**2/d**2
         

        
         
         
#         if tilt is None:
#         eulerAngles = rvec2euler(rvec)
#         tilt = np.clip(np.degrees(eulerAngles[1]),0,90)
#         print(tilt)
#         try:
#             angles, vals = getattr(emissivity_vs_angle, material)()
#         except AttributeError:
#             raise AttributeError("material[%s] is not in list of know materials: %s" %(
#                  material,[o[0] for o in getmembers(emissivity_vs_angle) 
#                            if isfunction(o[1])]))
#             
#         normEmissivity = np.clip(
#                             InterpolatedUnivariateSpline(
#                                     angles, vals)(tilt),0,1)
        return normEmissivity#, vf, normEmissivity, beta2, s
        
        
        


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
        
        if shape is None:
            s = self.img.shape
        else:
            s = shape

        # 1. DEFOCUS DUE TO DEPTH OF FIELD
        ################################## 
        depthMap = self.depthMap()
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
        wx = cv2.warpPerspective(np.asfarray(px), self.homography, 
                                 self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4  )
        wy = cv2.warpPerspective(np.asfarray(py), self.homography, 
                                 self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4  )

        pxSizeFactorX= (1/np.abs(np.gradient(wx)[1]))
        pxSizeFactorY= (1,np.abs(np.gradient(wy)[0]))

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
                    i = cv2.warpPerspective(i, self.homography, 
                                            self._newBorders,
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
#         if self.tvec is None:
        if self._pose is None:
            if self.quad is not None:
                self._pose = self._poseFromQuad()
            else:
                self._pose = self._poseFromHomography()
        return self._pose   


    def setPose(self, obj_center=None, distance=None, 
                   rotation=None, tilt=None, pitch=None):
        tvec,rvec = self.pose()
        
        if distance is not None:
            tvec[2,0]=distance
        if obj_center is not None:
            tvec[0,0]=obj_center[0]
            tvec[1,0]=obj_center[1]
        
        if rotation is None and tilt is None:
            return rvec
        r, t, p = rvec2euler(rvec)
        if rotation is not None:
            r = np.radians(rotation)
        if tilt is not None:
            t =  np.radians(tilt)
        if pitch is not None:
            p =   np.radians(pitch)
        rvec =  euler2rvec(r,t,p)        

        self._pose = tvec,rvec


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
        return self._poseFromQuad(quad)
        

    def objCenter(self):
        if self.quad is None:
            sy,sx = self.img.shape[:2]
            return sx//2, sy//2
        return self.quad[:,0].mean(), self.quad[:,1].mean()
            
        
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
        img_pn = np.ascontiguousarray(quad[:,:2], 
                        dtype=np.float32).reshape((4,1,2))

        obj_pn = self.obj_points - self.obj_points.mean(axis=0)

        retval, rvec, tvec = cv2.solvePnP(
                    obj_pn, 
                    img_pn, 
                    self.opts['cameraMatrix'], 
                    self.opts['distCoeffs'],
                    flags=cv2.SOLVEPNP_P3P #because exactly four points are given 
                    )
        if retval is None:
            print("Couln't estimate pose")
        return tvec, rvec


    @property
    def obj_points(self):
        if self._obj_points is None:

            if self.refQuad is not None:
                quad = self.refQuad
            else:
                quad = self.quad
            #                     raise AttributeError
            #                 try: quad = self.refQuad
            #                 except AttributeError: quad = self.quad
            
            #GET HOMOGRAPHIE FROM QUAD
#             fixedX,fixedY = None,None
            #try:
                #new size is given
#                 sx, sy = self.opts['new_size']
#                 if sx is None:
#                     if sy is not None:
#                         fixedY = sy
#                     raise TypeError()
#                 if sy is None:
#                     if sx is not None:
#                         fixedX = sx
#                 else:
                    #raise TypeError()
            #except TypeError:
            try:
                #estimate size
                sx = self.opts['obj_width_mm']
                sy = self.opts['obj_height_mm']
                aspectRatio = sx/sy
            except TypeError:
                aspectRatio = calcAspectRatioFromCorners(quad, 
                                                        self.opts['in_plane'])
                print('aspect ratio assumed to be %s' %aspectRatio)
                    #new image border keeping aspect ratio
#                     if fixedX or fixedY:
#                         if fixedX:
#                             sx = fixedX
#                             sy = sx/aspectRatio
#                         else:
#                             sy = fixedY
#                             sx = sy*aspectRatio
#                     else:   
                                 

#                 print (ssx/sx, ssy/sy)
#                 self.px_per_phys_unit = 0.5 * (ssx/sx + ssy/sy)
#                 print(self.px_per_phys_unit,77777777)

            #output size:
            if None in self._newBorders:
                ssx,ssy = self._calcQuadSize(quad, aspectRatio)
                bx, by = self._newBorders
                if bx is None:
                    bx = int(round(ssx))
                if by is None:
                    by = int(round(ssy))
                self._newBorders = (bx,by)
            
#             if self.refQuad is not None:
#                 ###cannot be right!
#                 raise Exception
#                 self._obj_points = self.refQuad.astype(np.float32)
#             else:


            
#             c = self.opts['cameraMatrix']
#             cx,cy = c[0,2], c[1,2]

            if None in (sx,sy):
                sx,sy = self._newBorders#ssx,ssy

#             hsx = sx/2
#             hsy = sy/2
#             self._obj_points = np.float32([
#                         [-hsx,  -hsy, 0],
#                         [ hsx,  -hsy, 0],
#                         [ hsx,   hsy, 0],
#                         [-hsx,   hsy, 0]])


            #image edges:
            self._obj_points = np.float32([
                        [0,  0, 0],
                        [sx, 0, 0],
                        [sx, sy, 0],
                        [0,  sy, 0]])
#             self._obj_points[:,0]+=cx
#             self._obj_points[:,1]+=cy
            
        return self._obj_points

#         h,w = self.opts['shape']#self.img.shape[:2]
# #         h = self.opts['obj_height_mm']
# #         w = self.opts['obj_width_mm']
# #         if w is None or h is None:
# #             w,h = 100,100
#         return np.array([
#                     [0, 0, 0 ],
#                     [w, 0, 0],
#                     [w, h, 0],
#                     [0, h, 0],
#                     ],dtype=np.float32)


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
    
        mx = int(img.max())
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
            y = x_length/ aspectRatio
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
    import sys
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
                               do_correctIntensity=False,
                               obj_height_mm=sy,
                               obj_width_mm=sx)
    pc.setReference(obj_corners)
    img2 = img.copy()
    pc.drawQuad(img2, thickness=2)
 
    #2. DISTORT THE IMAGE:
    #dist = img2.copy()#pc.distort(img2, rotX=10, rotY=20)
    dist = pc.distort(img2, rotX=10, rotY=20)
    dist2 = dist.copy()
    pc.draw3dCoordAxis(dist2, thickness=2)
    pc.drawQuad(dist2, thickness=2)
    
    
    p = pc.pose()#obj_center=None, distance=100, 
                #  rotation=0, tilt=30)

    depth = pc.corrDepthMap(img2.shape[:2], pose=p)#, distance=0)
    
    bg = dist[:,:,0]<10
    
#     import pylab as plt
#     print(dist.shape)
#     plt.imshow(dist[:,:,0]<10)
#     plt.show()
    depth[bg]=depth.mean()
    print('depth min:%s, max:%s' %(depth.min(), depth.max()))
    
    #3a. CORRECT WITH QUAD:
    corr = pc.correct(dist)
 
    #3b. CORRECT WITH WITHEN REFERENCE IMAGE
    #TODO:BROKEN IN OPENCV3.1 RELEASE##
#     pc.img = dist
#     pc.setReference(img)
#     corr2 = pc.correct(dist)
##################

    if 'no_window' not in sys.argv:
        cv2.imshow("1. original", img2)
        cv2.imshow("2. distorted", toUIntArray(dist2, dtype=np.uint8))
        cv2.imshow("3a. corr. with quad", toUIntArray(corr, dtype=np.uint8))
        cv2.imshow("depth", toUIntArray(depth, cutHigh=False, dtype=np.uint8) )

#         cv2.imshow("3b. corr. with ref. image", toUIntArray(corr2, dtype=np.uint8))
        cv2.waitKey(0)
    
