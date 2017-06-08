# coding=utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.linalg import norm

import cv2

from numba import jit

from scipy.ndimage.filters import maximum_filter
from scipy.optimize import curve_fit
from scipy.ndimage.interpolation import map_coordinates

from fancytools.math.boundingBox import boundingBox

# local
from imgProcessor.imgIO import imread
from imgProcessor.measure.sharpness._base import SharpnessBase
from imgProcessor.transformations import toUIntArray
from scipy.ndimage.measurements import center_of_mass


@jit(nopython=True)
def _findPoints(img, thresh, min_dist, points):
    gx = img.shape[0]
    gy = img.shape[1]
    px = 0
    n = 0
    l = len(points)
    for i in range(gx):
        for j in range(gy):
            px = img[i, j]

            if px > thresh:
                if n == l:
                    return
                points[n, 0] = j
                points[n, 1] = i
                n += 1
                # get kernel boundaries:
                xmn = i - min_dist
                if xmn < 0:
                    xmn = 0
                xmx = i + min_dist
                if xmx > gx:
                    xmx = gx

                ymn = j - min_dist
                if ymn < 0:
                    ymn = 0
                ymx = j + min_dist
                if ymx > gy:
                    ymx = gy
                # set surrounding area to zero
                # to ignore it
                for ii in range(xmx - xmn):
                    for jj in range(ymx - ymn):
                        img[xmn + ii, ymn + jj] = 0


class SharpnessfromPointSources(SharpnessBase):

    def __init__(self, min_dist=None, max_kernel_size=51,
                 max_points=3000, calc_std=False):
        #         self.n_points = 0
        self.max_points = max_points
        self.calc_std = calc_std
        # ensure odd number:
        self.kernel_size = k = max_kernel_size // 2 * 2 + 1

        if min_dist is None:
            min_dist = max_kernel_size // 2 + 10
        self.min_dist = min_dist

        self._psf = np.zeros(shape=(k, k))
    def addImg(self, img, roi=None):
        '''
        img - background, flat field, ste corrected image
        roi - [(x1,y1),...,(x4,y4)] -  boundaries where points are
        '''
        self.img = imread(img, 'gray')
        s0, s1 = self.img.shape

        if roi is None:
            roi = ((0, 0), (s0, 0), (s0, s1), (0, s1))

        k = self.kernel_size
        hk = k // 2

        # mask image
        img2 = self.img.copy()  # .astype(int)

        mask = np.zeros(self.img.shape)
        cv2.fillConvexPoly(mask, np.asarray(roi, dtype=np.int32), color=1)
        mask = mask.astype(bool)
        im = img2[mask]

        bg = im.mean()  # assume image average with in roi == background
        mask = ~mask
        img2[mask] = -1

        # find points from local maxima:
        self.points = np.zeros(shape=(self.max_points, 2), dtype=int)
        thresh = 0.8 * bg + 0.2 * im.max()

        _findPoints(img2, thresh, self.min_dist, self.points)
        self.points = self.points[:np.argmin(self.points, axis=0)[0]]

        # correct point position, to that every point is over max value:
        for n, p in enumerate(self.points):
            sub = self.img[p[1] - hk:p[1] + hk + 1, p[0] - hk:p[0] + hk + 1]
            i, j = np.unravel_index(np.nanargmax(sub), sub.shape)
            self.points[n] += [j - hk, i - hk]

        # remove points that are too close to their neighbour or the border
        mask = maximum_filter(mask, hk)
        i = np.ones(self.points.shape[0], dtype=bool)
        for n, p in enumerate(self.points):
            if mask[p[1], p[0]]:  # too close to border
                i[n] = False
            else:
                # too close to other points
                for pp in self.points[n + 1:]:
                    if norm(p - pp) < hk + 1:
                        i[n] = False
        isum = i.sum()
        ll = len(i) - isum
        print('found %s points' % isum)
        if ll:
            print(
                'removed %s points (too close to border or other points)' %
                ll)
            self.points = self.points[i]

#         self.n_points += len(self.points)

        # for finding best peak position:
#         def fn(xy,cx,cy):#par
#             (x,y) = xy
#             return 1-(((x-cx)**2 + (y-cy)**2)*(1/8)).flatten()

#         x,y = np.mgrid[-2:3,-2:3]
#         x = x.flatten()
#         y = y.flatten()
        # for shifting peak:
        xx, yy = np.mgrid[0:k, 0:k]
        xx = xx.astype(float)
        yy = yy.astype(float)

        self.subs = []


#         import pylab as plt
#         plt.figure(20)
#         img = self.drawPoints()
#         plt.imshow(img, interpolation='none')
# #                 plt.figure(21)
# #                 plt.imshow(sub2, interpolation='none')
#         plt.show()

        #thresh = 0.8*bg + 0.1*im.max()
        for i, p in enumerate(self.points):
            sub = self.img[p[1] - hk:p[1] + hk + 1,
                           p[0] - hk:p[0] + hk + 1].astype(float)
            sub2 = sub.copy()

            mean = sub2.mean()
            mx = sub2.max()
            sub2[sub2 < 0.5 * (mean + mx)] = 0  # only select peak
            try:
                # SHIFT SUB ARRAY to align peak maximum exactly in middle:
                    # only eval a 5x5 array in middle of sub:
                # peak = sub[hk-3:hk+4,hk-3:hk+4]#.copy()

                #                 peak -= peak.min()
                #                 peak/=peak.max()
                #                 peak = peak.flatten()
                    # fit paraboloid to get shift in x,y:
                #                 p, _ = curve_fit(fn, (x,y), peak, (0,0))
                c0, c1 = center_of_mass(sub2)

#                 print (p,c0,c1,hk)

                #coords = np.array([xx+p[0],yy+p[1]])
                coords = np.array([xx + (c0 - hk), yy + (c1 - hk)])

                #print (c0,c1)

                #import pylab as plt
                #plt.imshow(sub2, interpolation='none')

                # shift array:
                sub = map_coordinates(sub, coords,
                                      mode='nearest').reshape(k, k)
                # plt.figure(2)
                #plt.imshow(sub, interpolation='none')
                # plt.show()

                #normalize:
                bg = 0.25* (  sub[0].mean()   + sub[-1].mean() 
                            + sub[:,0].mean() + sub[:,-1].mean())
                 
                sub-=bg
                sub /= sub.max()

#                 import pylab as plt
#                 plt.figure(20)
#                 plt.imshow(sub, interpolation='none')
# #                 plt.figure(21)
# #                 plt.imshow(sub2, interpolation='none')
#                 plt.show()

                self._psf += sub

                if self.calc_std:
                    self.subs.append(sub)
            except ValueError:
                pass #sub.shape == (0,0)
            

    def intermetidatePSF(self, n=5, steps=None):
        s0,s1 = self._psf.shape
        if steps is not None:
            n = len(steps)
        else:
            steps = np.linspace(1,len(self.subs)-1,n, dtype=int)
        ipsf = np.empty((n,s0,s1))
        for o, j in enumerate(steps):
            ipsf[o] = np.mean(self.subs[:j], axis=0)
            ipsf[o] /= ipsf[o].sum()
        return ipsf, steps


    def std(self, i=None, filter_below=1.0, ref_psf=None):
        if i is None:
            i = len(self.points)
#         p = self.psf(filter_below)
        s0, s1 = self.subs[0].shape
        
#         subs = np.array([s[s0,s1] for s in self.subs])
        subs = np.array(self.subs)
        ipsfs,_ = self.intermetidatePSF(steps=range(len(subs)))
#         np.save('sssss', ipsfs)
#         subs/=subs.sum(axis=(1,2))
#         for s in subs:
#             self._filter(s, filter_below)
#             s/=s.sum()
#         print p
#         print subs[0]
#         return subs
#         sp = ((subs-p)**2)
#         trend = [np.nan]
        trend = []
        if ref_psf is None:
            ref_psf = ipsfs[-1]
        for n in range(1,len(subs)):
            #RMSE per step
#             if n ==100:
#                 import pylab as plt
#                 plt.plot(ref_psf.sum(axis=0))
#                 plt.plot(ipsfs[n].sum(axis=0), 'o-')
#                 plt.plot(ipsfs[-1].sum(axis=0), 'o-')
# 
#                 plt.show()
            
            trend.append( ((ref_psf-ipsfs[n])**2).mean()**0.5 )
            
#         for n in range(2,len(subs)+1):
            
#             trend.append( ((1/(n-1)) * ( sp[:n].sum(axis=0)  
#                                           )**0.5).mean() )          
            
            #standard deviation per step (psf.sum()==1)
#             import pylab as plt
#             plt.plot(p.mean(axis=0))
#             plt.plot(sp[0].mean(axis=0))
#             
#             plt.show()
#             trend.append( ((1/(n-1)) * ( sp[:n].sum(axis=0)  
#                                           )**0.5).mean() )
        return np.array(trend), (None,None,None)

        
        stdmap = (1/(i-1)) * ( sp.sum(axis=0)  )**0.5
        stdmap = stdmap.sum(axis=0)
        p = p.sum(axis=0)
        return np.array(trend), (p - stdmap, p, p + stdmap)

    # TODO: move and unit in extra PSF filter file
    @staticmethod
    def _filter(arr, val):
        a = (arr[0, :], arr[1, :], arr[:, 0], arr[:, -1])
        m = np.mean([aa.mean() for aa in a])
        s = np.mean([aa.std() for aa in a])
        t = m + val * s
        arr -= t
        arr[arr < 0] = 0

    
    #TODO: remove because is already in module PSF
    def psf(self, correct_size=True, filter_below=0.00):
        p = self._psf.copy()
        # filter background oscillations
        if filter_below:
            self._filter(p, filter_below)
#             mn = p.argsort()[4:].mean()
#             mn +=filter_below*p.max()-mn
#             ibg = p<mn
#             p[ibg] = mn
#         else:
#             ibg = p < p.min()
        # decrease kernel size if possible
        if correct_size:
            b = boundingBox(p == 0)
            s = p.shape
            ix = min(b[0].start, s[0] - b[0].stop)
            iy = min(b[1].start, s[1] - b[1].stop)
            s0, s1 = self._shape = (slice(ix, s[0] - ix),
                                    slice(iy, s[1] - iy))
            p = p[s0, s1]

        # scale
#         p-=p.min()
        p /= p.sum()
        self._corrPsf = p
        return p


#     @staticmethod
#     def _fn(v,sx,sy,rho):
#         r = gaussian2d((v[0],v[1]), sx, sy, 0, 0, rho)
#         r/=r.sum()
#         return r

    def drawPoints(self, img=None):
        c = False
        if img is None:
            img = self.img.copy()
        elif img is False:
            img = np.zeros(self.img.shape)
            c = 1
        if not c:
            c = img.max() - 1
        for p in self.points:
            cv2.circle(img, (p[0], p[1]), self.kernel_size // 2, c)
        return img


if __name__ == '__main__':
    # TODO: generic example
    pass
#     from imgProcessor.imgIO import imread, imwrite, out
#     from skimage import restoration
#     from fancytools.os.PathStr import PathStr
#     from imgProcessor.zDenoise import zDenoise
#     from matplotlib import pyplot as plt
#
#
#     def correct(i1,i2,bg1,bg2):
#         folder = i1.dirname()
#         i1 = imread(i1)
#         i2 = imread(i2)
#         bg1 = imread(bg1)
#         bg2 = imread(bg2)
#
#         i = zDenoise([i1,i2])[0]
#         bg = zDenoise([bg1,bg2])[0]
#         corr = i.astype(int)-bg
#         imwrite(folder.join('corrected.tif'), corr)
#
#
#     f = PathStr('C:\\Users\\elkb4\\Desktop\\PhD\\Measurements\\HuLC\\Calibration\\psfFromPoints')
#     f1 = f.join('1')
#     f2 = f.join('2')
#     f3 = f.join('3')
#
#     #imgs = f1.all()[1:]
#     #correct(imgs[2],imgs[3],imgs[0],imgs[1])
#     #imgs = f2.all()[1:]
#     #correct(imgs[2],imgs[3],imgs[0],imgs[1])
# #     imgs = f3.all()[1:]
# #     correct(imgs[2],imgs[3],imgs[0],imgs[1])
#
#     p = SharpnessfromPointSources()
#
#
# #     img = f1.join('corrected.tif')
# #     roi = [(1483,1353),
# #            (1781,1344),
# #            (1797,727),
# #            (1499,703)]
# #     p.addImg(img, roi)
# #
# #
#     img = f2.join('corrected.tif')
# #     roi = [(1083,1814),
# #            (1378,1817),
# #            (1358,1192),
# #            (1076,1180)]
# #     p.addImg(img, roi)
#
#
#     img = f3.join('corrected.tif')
#     roi = [(794,1870),
#            (2275,1874),
#            (2290,925),
#            (798,878)]
#     p.addImg(img, roi)
#     print 'found %s points' %p.n_points
#
#     psf = p.psf(filter_below=0.05)
#     print 'standard deviation: %s' %p.stdDev()
#     #p._std = 0.7
#     #psf = p.gaussianPsf()
#
#     np.save(f.join('psf.npy'), psf)
#
#
#
#     plt.imshow(psf, interpolation='none')
#     plt.colorbar()
#     plt.show()
#
#     img = p.drawPoints()
#     plt.imshow(img)
#     plt.colorbar()
#     plt.show()
#
#     #SHARPEN:
#     #img = imread(f2.join('corrected.tif'), 'gray', float)
#
#     img = imread('C:\\Users\\elkb4\Desktop\\PhD\Measurements\\EL round robin\\HULC el round robin\\mod7\\mod7_e180_g4_b1_V38-970_I7-801_T19-062_p2-2_n1__3.tif', 'gray', float)
#
#
#     mx = img.max()
#     img/=mx
#     #BEST:
#     deconvolved, _ = restoration.unsupervised_wiener(img, psf)
#
#     #FAST BUT STILL BLURRY:
#     #deconvolved = restoration.wiener(img, psf, balance=0.1)
#
#     #AS GOOD AS unsupervised, BUT SLOWER:
#     #deconvolved = restoration.richardson_lucy(img, psf,  iterations=4, clip=True)
#
#     deconvolved*=mx
#
#     imwrite(f.join('deblurred.tif'), deconvolved)
#     plt.imshow(deconvolved)
#     plt.colorbar()
#     plt.show()
