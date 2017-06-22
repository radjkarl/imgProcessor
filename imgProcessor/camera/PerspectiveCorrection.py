from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from transforms3d.euler import mat2euler, euler2mat
from inspect import getmembers, isfunction

from fancytools.math.Point3D import Point3D
from fancytools.math.vector3d import vectorAngle

from imgProcessor.imgIO import imread
from imgProcessor.utils.sortCorners import sortCorners
from imgProcessor.utils.genericCameraMatrix import genericCameraMatrix
from imgProcessor.equations.defocusThroughDepth import defocusThroughDepth
from imgProcessor.utils.imgPointToWorldCoord import imgPointToWorldCoord
from imgProcessor.features.PatternRecognition import PatternRecognition
from imgProcessor.utils.calcAspectRatioFromCorners import calcAspectRatioFromCorners
from imgProcessor.physics import emissivity_vs_angle

try:
    from PROimgProcessor.features.perspCorrectionViaQuad import perspCorrectionViaQuad
except ImportError:
    perspCorrectionViaQuad = None

# from imgProcessor.equations.vignetting import tiltFactor
# from imgProcessor.transform.simplePerspectiveTransform import simplePerspectiveTransform
# from imgProcessor.features.QuadDetection import QuadDetection
# from imgProcessor.utils.decompHomography import decompHomography


def rvec2euler(rvec):
    return mat2euler(cv2.Rodrigues(rvec)[0], axes='rzxy')


def euler2rvec(a0, a1, a2):
    return cv2.Rodrigues(euler2mat(a0, a1, a2, axes='rzxy'))[0]


class PerspectiveCorrection(object):

    def __init__(self,
                 img_shape,
                 # obj_height_mm=None,
                 # obj_width_mm=None,
                 cameraMatrix=None,
                 distCoeffs=np.zeros((5, 1)),
                 do_correctIntensity=False,
                 px_per_phys_unit=None,
                 new_size=(None, None),
                 in_plane=False,
                 border=0,
                 maxShear=0.05,
                 material='EL_Si_module',
                 cv2_opts={}):
        '''
        correction(warp+intensity factor) + uncertainty due to perspective distortion

        new_size = (sizey, sizex)
            if either sizey or sizex is None the resulting size will set 
            using an (assumed) aspect ratio 

        in_plane=True --> object has to tilt, only rotation and translation


        !!!
        given images need to be already free from lens distortion
        and distCoeffs should be 0
        !!!
        '''
        # TODO: remove camera matrix and dist coeffs completely - don't needed
        # since img needs to be lens corrected anyway

        # TODO: insert aspect ratio and remove obj_width, height
        self.opts = {  # 'obj_height_mm': obj_height_mm,
            #'obj_width_mm': obj_width_mm,
            'distCoeffs': distCoeffs.astype(np.float32),
            'do_correctIntensity': do_correctIntensity,
            'new_size': new_size,
            'in_plane': in_plane,
            'cv2_opts': cv2_opts,
            'border': border,
            'material': material,
            'maxShear': maxShear,
            'shape': img_shape[:2]}
        if cameraMatrix is None:
            cameraMatrix = genericCameraMatrix(img_shape)

        self.opts['cameraMatrix'] = cameraMatrix.astype(np.float32)
        self.refQuad = None
        self._obj_points = None
        self.px_per_phys_unit = px_per_phys_unit

        self._newBorders = self.opts['new_size']

    def setReferenceQuad(self, refQuad):
        '''TODO'''
        self.refQuad = sortCorners(refQuad)

    def setReference(self, ref):
        '''
        ref  ... either quad, grid, homography or reference image

        quad --> list of four image points(x,y) marking the edges of the quad
               to correct
        homography --> h. matrix to correct perspective distortion
        referenceImage --> image of same object without perspective distortion
        '''
#         self.maps = {}
        self.quad = None
#         self.refQuad = None
        self._camera_position = None
        self._homography = None
        self._homography_is_fixed = True
#         self.tvec, self.rvec = None, None
        self._pose = None

        # evaluate input:
        if isinstance(ref, np.ndarray) and ref.shape == (3, 3):
            # REF IS HOMOGRAPHY
            self._homography = ref
            # REF IS QUAD
        elif len(ref) == 4:
            self.quad = sortCorners(ref)

            # TODO: cleanup # only need to call once - here
            o = self.obj_points  # no property any more

            # REF IS IMAGE
        else:
            self.ref = imread(ref)
#             self._refshape = ref.shape[:2]
            self.pattern = PatternRecognition(self.ref)
            self._homography_is_fixed = False

    @property
    def homography(self):
        if self._homography is None:
            b = self.opts['border']
            if self.quad is not None:

                if self.refQuad is not None:
                    dst = self.refQuad.astype(np.float32)
                else:
                    sy, sx = self._newBorders
                    dst = np.float32([
                        [b,  b],
                        [sx - b, b],
                        [sx - b, sy - b],
                        [b,  sy - b]])

                self._homography = cv2.getPerspectiveTransform(
                    self.quad.astype(np.float32), dst)
            else:
                try:
                    # GET HOMOGRAPHY FROM REFERENCE IMAGE USING PATTERN
                    # RECOGNITION
                    self._Hinv = h = self.pattern.findHomography(self.img)[0]
                    H = self.pattern.invertHomography(h)
                except Exception as e:
                    print(e)
                    if perspCorrectionViaQuad:
                        # PROPRIETARY FALLBACK METHOD
                        quad = perspCorrectionViaQuad(
                            self.img, self.ref, border=b)
                        sy, sx = self.ref.shape
                        dst = np.float32([
                            [b,  b],
                            [sx - b, b],
                            [sx - b, sy - b],
                            [b,  sy - b]])

                        H = cv2.getPerspectiveTransform(
                            quad.astype(np.float32), dst)

                    else:
                        raise e

# #                 #test fit quality:
#                 if abs(decompHomography(H)[-1]) > self.opts['maxShear']:
#                     #shear too big
#

                self._homography = H

                sy, sx = self.opts['new_size']
                ssy, ssx = self.ref.shape[:2]
                if sx is None:
                    sx = ssx
                if sy is None:
                    sy = ssy
                self._newBorders = (sy, sx)

        return self._homography

    def distort(self, img, rotX=0, rotY=0, quad=None):
        '''
        Apply perspective distortion ion self.img
        angles are in DEG and need to be positive to fit into image

        '''
        self.img = imread(img)
        # fit old image to self.quad:
        corr = self.correct(self.img)

        s = self.img.shape
        if quad is None:
            wquad = (self.quad - self.quad.mean(axis=0)).astype(float)

            win_width = s[1]
            win_height = s[0]
            # project quad:
            for n, q in enumerate(wquad):
                p = Point3D(q[0], q[1], 0).rotateX(-rotX).rotateY(-rotY)
                p = p.project(win_width, win_height, s[1], s[1])
                wquad[n] = (p.x, p.y)
            wquad = sortCorners(wquad)
            # scale result so that longest side of quad and wquad are equal
            w = wquad[:, 0].max() - wquad[:, 0].min()
            h = wquad[:, 1].max() - wquad[:, 1].min()
            scale = min(s[1] / w, s[0] / h)
            # scale:
            wquad = (wquad * scale).astype(int)
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
        ], dtype=np.float32)

        homography = cv2.getPerspectiveTransform(
            wquad.astype(np.float32), objP)
        # distort corr:
        w = wquad[:, 0].max() - wquad[:, 0].min()
        h = wquad[:, 1].max() - wquad[:, 1].min()
        #(int(w),int(h))
        dist = cv2.warpPerspective(corr, homography, (int(w), int(h)),
                                   flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

        # move middle of dist to middle of the old quad:
        bg = np.zeros(shape=s)
        rmn = (bg.shape[0] / 2, bg.shape[1] / 2)

        ss = dist.shape
        mn = (ss[0] / 2, ss[1] / 2)  # wquad.mean(axis=0)
        ref = (int(rmn[0] - mn[0]), int(rmn[1] - mn[1]))

        bg[ref[0]:ss[0] + ref[0], ref[1]:ss[1] + ref[1]] = dist

        # finally move quad into right position:
        self.quad = wquad
        self.quad += (ref[1], ref[0])
        self.img = bg
        self._homography = None
        self._poseFromQuad()

        if self.opts['do_correctIntensity']:
            tf = self.tiltFactor()
            if self.img.ndim == 3:
                for col in range(self.img.shape[2]):
                    self.img[..., col] *= tf
            else:
                #                 tf = np.tile(tf, (1,1,self.img.shape[2]))
                self.img = self.img * tf

        return self.img

    def objectOrientation(self):
        tvec, r = self.pose()
        eulerAngles = mat2euler(cv2.Rodrigues(r)[0], axes='rzxy')

        tilt = eulerAngles[1]
        rot = eulerAngles[0]
        dist = tvec[2, 0]  # only take depth component np.linalg.norm(tvec)
        return dist, tilt, rot

    def correctGrid(self, img, grid):
        '''
        grid -> array of polylines=((p0x,p0y),(p1x,p1y),,,)
        '''

        self.img = imread(img)
        h = self.homography  # TODO: cleanup only needed to get newBorder attr.

        if self.opts['do_correctIntensity']:
            self.img = self.img / self._getTiltFactor(self.img.shape)

        s0, s1 = grid.shape[:2]
        n0, n1 = s0 - 1, s1 - 1

        snew = self._newBorders
        b = self.opts['border']

        sx, sy = (snew[0] - 2 * b) // n0, (snew[1] - 2 * b) // n1

        out = np.empty(snew[::-1], dtype=self.img.dtype)

        def warp(ix, iy, objP, outcut):
            shape = outcut.shape[::-1]
            quad = grid[ix:ix + 2,
                        iy:iy + 2].reshape(4, 2)[np.array([0, 2, 3, 1])]
            hcell = cv2.getPerspectiveTransform(
                quad.astype(np.float32), objP)
            cv2.warpPerspective(self.img, hcell, shape, outcut,
                                flags=cv2.INTER_LANCZOS4,
                                **self.opts['cv2_opts'])
            return quad

        objP = np.array([[0, 0],
                         [sx, 0],
                         [sx, sy],
                         [0, sy]], dtype=np.float32)
        # INNER CELLS
        for ix in range(1, n0 - 1):
            for iy in range(1, n1 - 1):
                sub = out[iy * sy + b: (iy + 1) * sy + b,
                          ix * sx + b: (ix + 1) * sx + b]
#                 warp(ix, iy, objP, sub)

                shape = sub.shape[::-1]
                quad = grid[ix:ix + 2,
                            iy:iy + 2].reshape(4, 2)[np.array([0, 2, 3, 1])]
#                 print(quad, objP)

                hcell = cv2.getPerspectiveTransform(
                    quad.astype(np.float32), objP)
                cv2.warpPerspective(self.img, hcell, shape, sub,
                                    flags=cv2.INTER_LANCZOS4,
                                    **self.opts['cv2_opts'])

#         return out
        # TOP CELLS
        objP[:, 1] += b
        for ix in range(1, n0 - 1):
            warp(ix, 0, objP, out[: sy + b,
                                  ix * sx + b: (ix + 1) * sx + b])
        # BOTTOM CELLS
        objP[:, 1] -= b
        for ix in range(1, n0 - 1):
            iy = (n1 - 1)
            y = iy * sy + b
            x = ix * sx + b
            warp(ix, iy, objP, out[y: y + sy + b, x: x + sx])
        # LEFT CELLS
        objP[:, 0] += b
        for iy in range(1, n1 - 1):
            y = iy * sy + b
            warp(0, iy, objP, out[y: y + sy, : sx + b])
        # RIGHT CELLS
        objP[:, 0] -= b
        ix = (n0 - 1)
        x = ix * sx + b
        for iy in range(1, n1 - 1):
            y = iy * sy + b
            warp(ix, iy, objP, out[y: y + sy, x: x + sx + b])
        # BOTTOM RIGHT CORNER
        warp(n0 - 1, n1 - 1, objP, out[-sy - b - 1:, x: x + sx + b])
#         #TOP LEFT CORNER
        objP += (b, b)
        warp(0, 0, objP, out[0: sy + b, 0: sx + b])
        # TOP RIGHT CORNER
        objP[:, 0] -= b
#         x = (n0-1)*sx+b
        warp(n0 - 1, 0, objP, out[: sy + b, x: x + sx + b])
#         #BOTTOM LEFT CORNER
        objP += (b, -b)
        warp(0, n1 - 1, objP, out[-sy - b - 1:, : sx + b])
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
        print("CORRECT PERSPECTIVE ...")
        self.img = imread(img)

        if not self._homography_is_fixed:
            self._homography = None
        h = self.homography

        if self.opts['do_correctIntensity']:
            tf = self.tiltFactor()
            self.img = np.asfarray(self.img)
            if self.img.ndim == 3:
                for col in range(self.img.shape[2]):
                    self.img[..., col] /= tf
            else:
                self.img = self.img / tf
        warped = cv2.warpPerspective(self.img,
                                     h,
                                     self._newBorders[::-1],
                                     flags=cv2.INTER_LANCZOS4,
                                     **self.opts['cv2_opts'])
        return warped

    def correctPoints(self, pts):
        if not self._homography_is_fixed:
            self._homography = None
        h = self._homography
        if pts.ndim == 2:
            pts = pts.reshape(1, *pts.shape)
        return cv2.perspectiveTransform(pts.astype(np.float32), h)

    @property
    def camera_position(self, pose=None):
        '''
        returns camera position in world coordinates using self.rvec and self.tvec
        from http://stackoverflow.com/questions/14515200/python-opencv-solvepnp-yields-wrong-translation-vector
        '''
        if pose is None:
            pose = self.pose()
        t, r = pose
        return -np.matrix(cv2.Rodrigues(r)[0]).T * np.matrix(t)

    def planeSfN(self, rvec):
        # get z:
            # 1 undistort plane:
        rot = cv2.Rodrigues(rvec)[0]
        aa = np.array([0., 0., 1.])
        return aa.dot(rot)

    def depthMap(self, midpointdepth=None, pose=None):
        shape = self.opts['shape']
        if pose is None:
            pose = self.pose()
        t, r = pose

        n = self.planeSfN(r)
        # z component from plane-equation solved for z:
        zpart = np.fromfunction(lambda y, x: (-n[0] * x
                                              - n[1] * y) / (
            -n[2]), shape)

        ox, oy = self.objCenter()
        v = zpart[int(oy), int(ox)]

        if midpointdepth is None:
            # TODO: review
            midpointdepth = t[2, 0]

        zpart += midpointdepth - v
        return zpart

    def cam2PlaneVectorField(self, midpointdepth=None, **kwargs):
        t, r = self.pose()
        shape = self.opts['shape']

        cam = self.opts['cameraMatrix']
        # move reference point from top left quad corner to
        # optical center:
#         q0 = self.quad[0]
        q0 = self.objCenter()
#         dx,dy = cam[0,2]-q0[0], cam[1,2]-q0[1]
        dx, dy = shape[1] // 2 - q0[0], shape[0] // 2 - q0[1]

        # x,y component of undist plane:
        rot0 = np.array([0, 0, 0], dtype=float)
        worldCoord = np.fromfunction(lambda x, y:
                                     imgPointToWorldCoord((y - dy, x - dx), rot0, t, cam
                                                          ), shape).reshape(3, *shape)
        # z component from plane-equation solved for z:
        n = self.planeSfN(r)
        x, y = worldCoord[:2]
        zpart = (-n[0] * x - n[1] * y) / (-n[2])
        ox, oy = self.objCenter()
        v = zpart[int(oy), int(ox)]

        if midpointdepth is None:
            # TODO: review
            midpointdepth = t[2, 0]
        zpart += midpointdepth - v
        worldCoord[2] = zpart
        return worldCoord

    # BEFORE REMOVING THINGS: MAKE EXTRA FN
    def viewAngle(self, **kwargs):
        '''
        calculate view factor between one small and one finite surface
        vf =1/pi * integral(cos(beta1)*cos(beta2)/s**2) * dA
        according to VDI heatatlas 2010 p961
        '''
        v0 = self.cam2PlaneVectorField(**kwargs)
        # obj cannot be behind camera
        v0[2][v0[2] < 0] = np.nan

        _t, r = self.pose()
        n = self.planeSfN(r)
        # because of different x,y orientation:
        n[2] *= -1
#         beta2 = vectorAngle(v0, vectorToField(n) )
        beta2 = vectorAngle(v0, n)
        return beta2

    def foreground(self, quad=None):
        '''return foreground (quad) mask'''
        fg = np.zeros(shape=self._newBorders[::-1], dtype=np.uint8)
        if quad is None:
            quad = self.quad
        else:
            quad = quad.astype(np.int32)
        cv2.fillConvexPoly(fg, quad, 1)
        return fg.astype(bool)

    def tiltFactor(self, midpointdepth=None,
                   printAvAngle=False):
        '''
        get tilt factor from inverse distance law
        https://en.wikipedia.org/wiki/Inverse-square_law
        '''
        # TODO: can also be only def. with FOV, rot, tilt
        beta2 = self.viewAngle(midpointdepth=midpointdepth)
        try:
            angles, vals = getattr(
                emissivity_vs_angle, self.opts['material'])()
        except AttributeError:
            raise AttributeError("material[%s] is not in list of know materials: %s" % (
                self.opts['material'], [o[0] for o in getmembers(emissivity_vs_angle)
                                        if isfunction(o[1])]))
        if printAvAngle:
            avg_angle = beta2[self.foreground()].mean()
            print('angle: %s DEG' % np.degrees(avg_angle))

        # use averaged angle instead of beta2 to not overemphasize correction
        normEmissivity = np.clip(
            InterpolatedUnivariateSpline(
                np.radians(angles), vals)(beta2), 0, 1)
        return normEmissivity

    @property
    def areaRatio(self):
        # AREA RATIO AFTER/BEFORE:
            # AREA OF QUADRILATERAL:
        if self.quad is None:
            q = self.quadFromH()[0]
        else:
            q = self.quad
        quad_size = 0.5 * abs((q[2, 0] - q[0, 0]) * (q[3, 1] - q[1, 1]) +
                              (q[3, 0] - q[1, 0]) * (q[0, 1] - q[2, 1]))
        sx, sy = self._newBorders

        return (sx * sy) / quad_size

    def standardUncertainties(self, focal_Length_mm, f_number, midpointdepth=1000,
                              focusAtYX=None,
                              # sigma_best_focus=0,
                              # quad_pos_err=0,
                              shape=None,
                              uncertainties=(0, 0)):
        '''
        focusAtXY - image position with is in focus
            if not set it is assumed that the image middle is in focus
        sigma_best_focus - standard deviation of the PSF
                             within the best focus (default blur)
        uncertainties - contibutors for standard uncertainty
                        these need to be perspective transformed to fit the new 
                        image shape
        '''
        # TODO: consider quad_pos_error
        # (also influences intensity corr map)

        if shape is None:
            s = self.img.shape
        else:
            s = shape

        # 1. DEFOCUS DUE TO DEPTH OF FIELD
        ##################################
        depthMap = self.depthMap(midpointdepth)
        if focusAtYX is None:
            # assume image middle is in-focus:
            focusAtYX = s[0] // 2, s[1] // 2
        infocusDepth = depthMap[focusAtYX]
        depthOfField_blur = defocusThroughDepth(
            depthMap, infocusDepth, focal_Length_mm, f_number, k=2.335)

        # 2. INCREAASED PIXEL SIZE DUE TO INTERPOLATION BETWEEN
        #   PIXELS MOVED APARD
        ######################################################
        # index maps:
        py, px = np.mgrid[0:s[0], 0:s[1]]
        # warped index maps:
        wx = cv2.warpPerspective(np.asfarray(px), self.homography,
                                 self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4)
        wy = cv2.warpPerspective(np.asfarray(py), self.homography,
                                 self._newBorders,
                                 borderValue=np.nan,
                                 flags=cv2.INTER_LANCZOS4)

        pxSizeFactorX = 1 / np.abs(np.gradient(wx)[1])
        pxSizeFactorY = 1 / np.abs(np.gradient(wy)[0])

        # WARP ALL FIELD TO NEW PERSPECTIVE AND MULTIPLY WITH PXSIZE FACTOR:
        depthOfField_blur = cv2.warpPerspective(
            depthOfField_blur, self.homography, self._newBorders,
            borderValue=np.nan,
        )

        # perspective transform given uncertainties:
        warpedU = []
        for u in uncertainties:
            #             warpedU.append([])
            #             for i in u:
            # print i, type(i), isinstance(i, np.ndarray)
            if isinstance(u, np.ndarray) and u.size > 1:
                u = cv2.warpPerspective(u, self.homography,
                                        self._newBorders,
                                        borderValue=np.nan,
                                        flags=cv2.INTER_LANCZOS4)  # *f

            else:
                # multiply with area ratio: after/before perspective warp
                u *= self.areaRatio

            warpedU.append(u)

        # given uncertainties after warp:
        ux, uy = warpedU

        ux = pxSizeFactorX * (ux**2 + depthOfField_blur**2)**0.5
        uy = pxSizeFactorY * (uy**2 + depthOfField_blur**2)**0.5

        # TODO: remove depthOfField_blur,fx,fy from return
        return ux, uy, depthOfField_blur, pxSizeFactorX, pxSizeFactorY

    def pose(self):
        #         if self.tvec is None:
        #         if self._pose is None:
        if self.quad is not None:
            self._pose = self._poseFromQuad()
        else:
            self._pose = self._poseFromHomography()
        return self._pose

    def setPose(self, obj_center=None, distance=None,
                rotation=None, tilt=None, pitch=None):
        tvec, rvec = self.pose()

        if distance is not None:
            tvec[2, 0] = distance
        if obj_center is not None:
            tvec[0, 0] = obj_center[0]
            tvec[1, 0] = obj_center[1]

        if rotation is None and tilt is None:
            return rvec
        r, t, p = rvec2euler(rvec)
        if rotation is not None:
            r = np.radians(rotation)
        if tilt is not None:
            t = np.radians(tilt)
        if pitch is not None:
            p = np.radians(pitch)
        rvec = euler2rvec(r, t, p)

        self._pose = tvec, rvec

    def _poseFromHomography(self):
        quad = self.quadFromH()
        return self._poseFromQuad(quad)

    def quadFromH(self):
        sy, sx = self.img.shape[:2]
        # image edges:
        objP = np.array([[
            [0, 0],
            [sx, 0],
            [sx, sy],
            [0, sy],
        ]], dtype=np.float32)
        return cv2.perspectiveTransform(objP, self._Hinv)

    def objCenter(self):
        if self.quad is None:
            sy, sx = self.img.shape[:2]
            return sx // 2, sy // 2
        return self.quad[:, 0].mean(), self.quad[:, 1].mean()

    def _poseFromQuad(self, quad=None):
        '''
        estimate the pose of the object plane using quad
            setting:
        self.rvec -> rotation vector
        self.tvec -> translation vector
        '''
        if quad is None:
            quad = self.quad
        if quad.ndim == 3:
            quad = quad[0]
        # http://answers.opencv.org/question/1073/what-format-does-cv2solvepnp-use-for-points-in/
        # Find the rotation and translation vectors.
        img_pn = np.ascontiguousarray(quad[:, :2],
                                      dtype=np.float32).reshape((4, 1, 2))

        obj_pn = self.obj_points - self.obj_points.mean(axis=0)
        retval, rvec, tvec = cv2.solvePnP(
            obj_pn,
            img_pn,
            self.opts['cameraMatrix'],
            self.opts['distCoeffs'],
            flags=cv2.SOLVEPNP_P3P  # because exactly four points are given
        )
        if not retval:
            print("Couln't estimate pose")
        return tvec, rvec

    @property
    def obj_points(self):
        if self._obj_points is None:

            if self.refQuad is not None:
                quad = self.refQuad
            else:
                quad = self.quad

            try:
                # estimate size
                sy, sx = self.opts['new_size']
#                 sy = self.opts['obj_height_mm']
                aspectRatio = sx / sy
            except TypeError:
                aspectRatio = calcAspectRatioFromCorners(quad,
                                                         self.opts['in_plane'])
                print('aspect ratio assumed to be %s' % aspectRatio)
            # output size:
            if None in self._newBorders:
                b = self.opts['border']
                ssx, ssy = self._calcQuadSize(quad, aspectRatio)
                bx, by = self._newBorders
                if bx is None:
                    bx = int(round(ssx + 2 * b))
                if by is None:
                    by = int(round(ssy + 2 * b))
                self._newBorders = (bx, by)

            if None in (sx, sy):
                sx, sy = self._newBorders

            # image edges:
            self._obj_points = np.float32([
                [0,  0, 0],
                [sx, 0, 0],
                [sx, sy, 0],
                [0,  sy, 0]])

        return self._obj_points

    def drawQuad(self, img=None, quad=None, thickness=30):
        '''
        Draw the quad into given img 
        '''
        if img is None:
            img = self.img
        if quad is None:
            quad = self.quad
        q = np.int32(quad)
        c = int(img.max())
        cv2.line(img, tuple(q[0]), tuple(q[1]), c, thickness)
        cv2.line(img, tuple(q[1]), tuple(q[2]), c, thickness)
        cv2.line(img, tuple(q[2]), tuple(q[3]), c, thickness)
        cv2.line(img, tuple(q[3]), tuple(q[0]), c, thickness)
        return img

    def draw3dCoordAxis(self, img=None, thickness=8):
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
        # self.opts['obj_width_mm'], self.opts['obj_height_mm']
        w, h = self.opts['new_size']
        axis = np.float32([[0.5 * w, 0.5 * h, 0],
                           [w, 0.5 * h, 0],
                           [0.5 * w, h, 0],
                           [0.5 * w, 0.5 * h, -0.5 * w]])
        t, r = self.pose()
        imgpts = cv2.projectPoints(axis, r, t,
                                   self.opts['cameraMatrix'],
                                   self.opts['distCoeffs'])[0]

        mx = int(img.max())
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, mx), thickness)
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, mx, 0), thickness)
        cv2.line(
            img, origin, tuple(imgpts[3].ravel()), (mx, 0, 0), thickness * 2)
        return img

    @staticmethod
    def _calcQuadSize(corners, aspectRatio):
        '''
        return the size of a rectangle in perspective distortion in [px]
        DEBUG: PUT THAT BACK IN??::
            if aspectRatio is not given is will be determined
        '''
        if aspectRatio > 1:  # x is bigger -> reduce y
            x_length = PerspectiveCorrection._quadXLength(corners)
            y = x_length / aspectRatio
            return x_length, y
        else:  # y is bigger -> reduce x
            y_length = PerspectiveCorrection._quadYLength(corners)
            x = y_length * aspectRatio
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
        p0, p1 = line
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        return (dx**2 + dy**2)**0.5


if __name__ == '__main__':
    # in this example we will rotate a given image in perspective
    # and then correct the perspective with given corners or a reference
    #(the undistorted original) image
    import sys
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    from imgProcessor.transformations import toUIntArray

    # 1. LOAD TEST IMAGE
    path = PathStr(imgProcessor.__file__).dirname().join(
        'media', 'electroluminescence', 'EL_module_orig.PNG')
    img = imread(path)

    # DEFINE OBJECT CORNERS:
    sy, sx = img.shape[:2]
    # x,y
    obj_corners = [(8,  2),
                   (198, 6),
                   (198, 410),
                   (9,  411)]
    # LOAD CLASS:
    pc = PerspectiveCorrection(img.shape,
                               do_correctIntensity=True,
                               new_size=(sy, sx)
                               #                                obj_height_mm=sy,
                               #                                obj_width_mm=sx
                               )
    pc.setReference(obj_corners)
    img2 = img.copy()
    pc.drawQuad(img2, thickness=2)
    # 2. DISTORT THE IMAGE:
    # dist = img2.copy()#pc.distort(img2, rotX=10, rotY=20)
    dist = pc.distort(img2, rotX=10, rotY=20)
    dist2 = dist.copy()
    pc.draw3dCoordAxis(dist2, thickness=2)
    pc.drawQuad(dist2, thickness=2)

    depth = pc.depthMap()  # , distance=0)

    bg = dist[:, :, 0] < 10

    depth[bg] = depth.mean()
    print('depth min:%s, max:%s' % (depth.min(), depth.max()))

    # 3a. CORRECT WITH QUAD:
    corr = pc.correct(dist)

    # 3b. CORRECT WITH WITHEN REFERENCE IMAGE
    pc.img = dist
    pc.setReference(img)
    corr2 = pc.correct(dist)
##################

    if 'no_window' not in sys.argv:
        cv2.imshow("1. original", img2)
        cv2.imshow("2. distorted", toUIntArray(dist2, dtype=np.uint8))
        cv2.imshow("3a. corr. with quad", toUIntArray(corr, dtype=np.uint8))
        cv2.imshow("depth", toUIntArray(depth, cutHigh=False, dtype=np.uint8))
        cv2.imshow(
            "3b. corr. with ref. image", toUIntArray(corr2, dtype=np.uint8))
        cv2.waitKey(0)
