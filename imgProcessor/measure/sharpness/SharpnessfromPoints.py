import numpy as np
from numpy.linalg import norm

import cv2

from numba import jit

from scipy.ndimage.filters import maximum_filter
from scipy.optimize import curve_fit
from scipy.ndimage.interpolation import map_coordinates

from imgProcessor.imgIO import imread
from imgProcessor.equations.gaussian2d import gaussian2d

from fancytools.math.boundingBox import boundingBox

#local
from imgProcessor.measure.sharpness._base import SharpnessBase



@jit(nopython=True)
def _findPoints(img, thresh, min_dist, points):
    gx = img.shape[0]
    gy = img.shape[1]
    px = 0
    n = 0
    l = len(points)
    for i in xrange(gx):
        for j in xrange(gy):
            px = img[i,j]

            if px > thresh:
                if n == l:
                    return
                points[n,0] = j
                points[n,1] = i
                n+=1
                #get kernel boundaries:
                xmn = i-min_dist
                if xmn < 0:
                    xmn = 0
                xmx = i+min_dist
                if xmx > gx:
                    xmx = gx
                    
                ymn = j-min_dist
                if ymn < 0:
                    ymn = 0
                ymx = j+min_dist
                if ymx > gy:
                    ymx = gy
                #set surrounding area to zero
                #to ignore it
                for ii in xrange(xmx-xmn):
                    for jj in xrange(ymx-ymn):
                        img[xmn+ii,ymn+jj] = 0

    
            
class SharpnessfromPointSources(SharpnessBase):
    def __init__(self, min_dist=None, max_kernel_size=51, 
                 max_points=3000):
        self.n_points = 0
        self.max_points = max_points
        #ensure odd number:
        self.kernel_size = k = max_kernel_size//2*2+1
        
        if min_dist is None:
            min_dist = max_kernel_size//2+10
        self.min_dist = min_dist

        self._psf = np.zeros(shape=(k,k))


    def addImg(self, img, roi):
        '''
        img - background, flat field, ste corrected image
        roi - [(x1,y1),...,(x4,y4)] -  boundaries where points are
        '''
        self.img = imread(img, 'gray')

        k = self.kernel_size
        hk = k//2

        #mask image
        mask = np.zeros(self.img.shape)
        cv2.fillConvexPoly(mask, np.asarray(roi, dtype=np.int32), color=1)
        mask = mask.astype(bool)
        img2 = self.img.copy().astype(int)
        im = img2[mask]
        
        bg = im.mean()
        mask = ~mask
        img2[mask] = -1

        #find points from local maxima:
        self.points = np.zeros(shape=(self.max_points,2), dtype=int)
        thresh = 0.8*bg + 0.2*im.max()
        
        _findPoints(img2, thresh, self.min_dist, self.points)
        self.points = self.points[:np.argmin(self.points,axis=0)[0]]

        #remove points that are too close to their neighbour or the border
        mask = maximum_filter(mask, hk)
        i = np.ones(self.points.shape[0], dtype=bool)
        for n, p in enumerate(self.points):
            if mask[p[1],p[0]]: #too close to border
                i[n]=False
            else:
                for pp in self.points[n+1:]:
                    if norm(p-pp) < hk+1: 
                        i[n]=False
        isum = i.sum()
        ll = len(i)-isum
        print 'found %s points' %isum
        if ll:
            print 'removed %s points (too close to border or other points)' %ll
            self.points = self.points[i]
                 
        self.n_points += len(self.points)

        #for finding best peak position:
        def fn((x,y),cx,cy):#par
            return 1-(((x-cx)**2 + (y-cy)**2)*(1.0/8)).flatten() 
        
        x,y = np.mgrid[-2:3,-2:3]
        x = x.flatten()
        y = y.flatten()
        #for shifting peak:
        xx,yy = np.mgrid[0:k,0:k]
        xx = xx.astype(float)
        yy = yy.astype(float)

        for p in self.points:
            sub = self.img[p[1]-hk:p[1]+hk+1,p[0]-hk:p[0]+hk+1].astype(float)

            #SHIFT SUB ARRAY to align peak maximum exactly in middle:
                #only eval a 5x5 array in middle of sub:
            peak = sub[hk-2:hk+3,hk-2:hk+3].copy()
            peak -= peak.min()
            peak/=peak.max()
            peak = peak.flatten()
                #fit paraboloid to get shift in x,y:
            p, _ = curve_fit(fn, (x,y), peak, (0,0))
            coords = np.array([xx+p[0],yy+p[1]])
                #shift array:
            sub =  map_coordinates(sub, coords, 
                        mode='nearest').reshape(k,k)
            #normalize:
            sub-=bg
            sub /= sub.max()
            self._psf+=sub


    def psf(self, correct_size=True, filter_below=0.00):
        p = self._psf.copy()
        #filter background oscillations
        if filter_below:
            mn = p.argsort()[4:].mean()
            mn +=filter_below*p.max()-mn
            ibg = p<mn
            p[ibg] = mn
        else:
            ibg = p < p.min()
            
        #decrease kernel size if possible
        if correct_size:
            b = boundingBox(~ibg)
            s = p.shape
            ix = min(b[0].start, s[0]-b[0].stop)
            iy = min(b[1].start, s[1]-b[1].stop)
            p = p[ix:s[0]-ix,iy:s[1]-iy]
        
        #scale
        p-=p.min()
        p/=p.sum()
        self._corrPsf = p
        return p


    @staticmethod
    def _fn(v,sx,sy,rho):
        r = gaussian2d((v[0],v[1]), sx, sy, 0, 0, rho)
        r/=r.sum()
        return r


    def drawPoints(self, img=None):
        c = False
        if img is None:
            img = self.img.copy()
        elif img is False:
            img = np.zeros(self.img.shape)
            c = 1
        if not c:
            c = img.max()-1
        for p in self.points:
            cv2.circle(img, (p[0],p[1]), self.kernel_size//2, c)
        return img




if __name__ == '__main__':
    #TODO: generic example
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