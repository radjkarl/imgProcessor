from __future__ import division

import numpy as np

import cv2

from scipy.ndimage.filters import maximum_filter, minimum_filter

from fancytools.math import line as ln
# from fancytools.fit.polyFitIgnoringOutliers import polyFitIgnoringOutliers

from imgProcessor.camera.PerspectiveCorrection import PerspectiveCorrection
from imgProcessor.imgSignal import signalMinimum
from imgProcessor.imgIO import imread
from fancytools.math.line import angle2#, toFn
# from scipy.optimize.minpack import curve_fit
# from scipy.optimize import minimize

from imgProcessor.features.minimumLineInArray import minimumLineInArray
# from imgProcessor.filters.edgesFromBoolImg import edgesFromBoolImg

    
    
    
#TODO: these methods might be useful for later, dont delete...
# def rubustRegression(x,y):
# 
#     p = polyFitIgnoringOutliers(x, y, deg=1, niter=5, nstd=1)
#     # extract edge points:
#     x0, x1 = x[0], x[-1]
#     y0, y1 = p(x0), p(x1)
#     return x0,y0,x1,y1
# 
# 
# 
# def nClosest(x,y, r=0.3, npieces=3):
#     n = max(5,int(len(y)*r/npieces))
#     i = 0
#     di = len(y)/npieces
#     ind = []
#     for _ in range(npieces):
#         yi = y[i:i+di]
#         ind.extend(i+yi.argsort()[:n])
#         i+=di
#     
#     return rubustRegression(x[ind], y[ind])
# 
# 
# 
# 
# 
# def lineSum(x0,y0,x1,y1):
#     ln = int(((x1-x0)**2 + (y0-y1)**2)**0.5)
#     return img[np.linspace(y0,y1,ln, dtype=int), 
#         np.linspace(x0,x1,ln, dtype=int)].sum()
# 
# 
# def mkline(x0,y0,x1,y1):
#     ln = int(((x1-x0)**2 + (y0-y1)**2)**0.5)
#     return np.linspace(x0,x1,ln, dtype=int), np.linspace(y0,y1,ln, dtype=int)
# 
# 
# def evalLine(x,y,line0):
#     m,n = toFn(line0)[:2]
#     yfit = np.linspace(m*x[0]+n,m*x[-1]+n, len(x))
#     max_dev = 5
#     dy = np.abs(y-yfit)
#     return len(dy[dy<max_dev])
# 
# 
# 
# 
#     
# 
# def findLine(x,y):
# #     xx = np.arange(len(y))[np.isfinite(y)]
#     
#     
#     mx,mn = y.max(), y.min()
#     n, bins = np.histogram(y, range=(mn,mx),bins=mx-mn)
# 
#     import pylab as plt
#     plt.plot(n)
# #     plt.plot(xfit,yfit)
# #     plt.plot(xfit2,yfit2, 'o-')
# 
#     plt.show()  
#     
#     def fn(mn):
#         m,n = mn
#         print(m,n)
# #         import pylab as plt
# #         plt.plot(x,m*x+n)
# #         plt.plot(x,y)
# #     
# #         plt.show()
#         
#         yfit = m*x+n
#         
#         max_dev = 5
#         dy = np.abs(y-yfit)**0.5
#         return dy#.mean()
#         ll = max(1,len(dy[dy<max_dev]))
#         
#         return 1/ll
#         
#         
#         
#         return m*x+n
# 
#     print(minimize(fn, (0,0)))
# 
#     
# 
#     m,n = curve_fit(fn,x,y)[0]
#     
#     x0,y0,x1,y1 = x[0], m*x[0]+n, x[-1], x[-1]*m+n
#     return x0,y0,x1,y1
#     print(popt)
#     
#     
#     line0 = rubustRegression(x,y)
#     
#     score0 = evalLine(x,y,line0)
# 
#     line1 = nClosest(x,y)
#     xfit2,yfit2 = mkline(*line1)
#     
#     score1 = evalLine(x,y,line1)
#     
#     print(score0,score1)
#     
#     xfit,yfit = mkline(*line0)
#     
# 
#     
#     
# 
# 
#     import pylab as plt
#     plt.plot(x,y)
#     plt.plot(xfit,yfit)
#     plt.plot(xfit2,yfit2, 'o-')
# 
#     plt.show()
#     
#     
#     if score1 > score0:
#         return line1
#     return line0  
# 
# 
#     return line0
#     s0 = lineSum(img, *line0)
# 
#     line1 = nClosest(x,y)
#     s1 = lineSum(img, *line1)  
#     print(s0,s1)
#     if s0 > s1:
#         return line0
#     return line1 

def angle_between(l1, l2):
    return angle_between2((l1[2]-l1[0],l1[3]-l1[1]), (l2[2]-l2[0],l2[3]-l2[1]))


def angle_between2(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


#TODO: speeed up through 
#cutting image with bounding box.
#than findMinLine will be faster

class QuadDetection(object):
    '''
    detect the corners of a (bright) quadrilateral object in an image
    e.g. a PV cell/module in an EL image
    '''
#     rules = ('rubustRegression', 'distanceWeighted')
    def __init__(self, img=None, isThresh=False, isRect=False,#, rule=rubustRegression  # , vertices=None#, refinePositions=True
                 ):
        '''
        @param img -> input image
        @paramn vertices -> routh estimate of corner positions
        @refinePositions -> whether to refine (found) corner positions
        '''
        #remove?? all correction, because is in PerspCor anyway
        self._pc = None
        if not isThresh:
            self.img = imread(img, 'gray')
    #         self.vertices = vertices
    
    
            
    
    
            thresh = img > signalMinimum(img)
            
#         print(signalMinimum(img))
#         import pylab as plt
#         print(signalMinimum(img))
#         plt.imshow(img)
#         plt.colorbar()
#         plt.show()
        # remove small features:
            thresh = minimum_filter(thresh, 5)
            
            thresh = maximum_filter(thresh, 5)
        else:
            thresh = img
        self.thresh = thresh
        if isRect:
            cnts = cv2.findContours(thresh.astype(np.uint8), 
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
#             print(len(cnt), cnt,9999999)
#             print(cnt.shape,cnt.dtype )
#             cnt = sorted(cnt, key=cv2.contourArea, reverse = True)[0]
            areas = [cv2.contourArea(c) for c in cnts]
            minArea = (img.shape[0]*img.shape[1])/100
            fcnts = []
            for n, a in enumerate(areas):
                if a > minArea:
                    fcnts.extend(cnts[n])

            
#             print (cnt)
            rect = cv2.minAreaRect(np.array(fcnts))
#             box = cv2.boxPoints(rect)
            self.vertices = cv2.boxPoints(rect)
        else:
#             print(rect, box)



            lines = self._findQuadLines()
    
            self.vertices = self._verticesFromLines(lines)

#         if True:#isRect:
#             #if one corner is wrongly detected, it can be substituted, 
#             
#             a = self._angles(lines)
#             print(a)
#             return
#             #find faulty edge:
#             invalid = a<1.4
#             assert invalid.sum()<2
#             i = np.argmin(invalid)
#             v = self.vertices
#             v[i]= v[i-1]+v[i-2]+v[i-3]
            
#             #test plausability:
#             if a.min()<1.4:
#                 lines2 = self._findQuadLines(nClosest)    
#                 a2 = self._angles(lines2)
#                 print(a, a2)
#                 #take better one:
#                 #if np.abs(a2-np.pi).mean() < np.abs(a-np.pi).mean():
#                 lines = lines2 





    @staticmethod
    def _angles(lines):
        (ltop, lbottom, lleft, lright) = lines
        ll = (ltop, 
              lright, 
              lbottom, 
              lleft, 
              ltop)
        print(111,[angle_between(l0,l1) for l0,l1 in zip(ll[:-1], ll[1:])])

        
        return np.array([abs(angle2(l0,l1)) for l0,l1 in zip(ll[:-1], ll[1:])])
        

#         self.plot()

    def correct(self, img=None, **kwargs):
        '''
        correct perspective
        forwards kwargs to PerspectiveCorrection
        '''
        if img is None:
            img = self.img

        if self._pc:
            img = self._pc.correct(img)
            img = img[self._shape]
            return img
        else:
            self._pc = PerspectiveCorrection(img.shape, **kwargs)
            self._pc.setReference(self.vertices)
            img = self._pc.correct(img)

            self._corrected = True
            return img

#     @staticmethod
#     def _to8bitImg(img):
#         if img.dtype == np.uint8:
#             return img
#         r = signalRange(img)
#         return toUIntArray(img, dtype=np.uint8, range=r)




    @staticmethod
    def _findEdgeLine(img, axis=0, start=0, stop=None, direction=1):
        # find the approximate edge line of and object within an image
        # along a given axis
        s = img.shape
#         if start != 0 or stop != None:
            # cut image
        if direction == 1:
            cut = slice(start, stop)
        else:
            cut = slice(stop, start, -1)
        if axis == 0:
            img = img[cut, :]
        else:
            img = img[:, cut]

        sx = s[int(~axis)]

        x = np.arange(sx)
        y = np.argmax(img, axis=axis)

        #exclude borders:
        valid = np.logical_and(y!=0, y!=sx-1)
        y = y[valid]
        x = x[valid]
        
        #create array with that edge:
        s0,s1 = img.shape
        if axis==1:
            s0,s1=s1,s0
        arr = np.ones((s0,s1), dtype=bool)
        arr[y,x] = 0
        #brute force best line:  
        y0,y1 = minimumLineInArray(arr)
        x0,x1 = 0, sx
        
        
        

#         
#         
#         
# 
# #         ind = np.argmax(img, axis=axis)
# #         arr = np.zeros(img.shape)
# #         try:
# #             arr[x, ind]=-1
# #         except IndexError:
# #             arr[ind,x]=-1
# #         import pylab as plt
# #         plt.imshow(img)
# #         print(axis)
# #         plt.show()
#         
#         # return first non=zero value along given axis
#         y = np.argmax(img, axis=axis)
# #         print(y)
#         #exclude borders:
#         valid = np.logical_and(y!=0, y!=sx-1)
#         #if valid.sum() > 0.2 * sx:
#         y = y[valid]
#         x = x[valid]


#         valid = np.abs(np.diff(y))<10
#         print valid, valid.sum()
#         valid = np.insert(valid,0,1)
#         y = y[valid]
#         x = x[valid]
#
#         from sklearn.linear_model import TheilSenRegressor
#         ts = TheilSenRegressor()
#         X = x[:, np.newaxis]
#         ts.fit(X,y)
#         p = ts.predict
#         x0,x1 = x[0], x[-1]
#         y0,y1 = p(np.array((x0,x1)).reshape(2,1))

        # filter outliers:

#         x0, y0, x1, y1 = findLine(x,y)#findLine(x,y)

        
        

#         print(y)




        if axis == 1:
            x0, x1, y0, y1 = y0, y1, x0, x1
        # move points to actual position in image
        if direction == 1:
            if start != 0:
                if axis == 0:
                    y0, y1 = y0 + start, y1 + start
                else:
                    x0, x1 = x0 + start, x1 + start
        else:
            if stop == None:
                stop = s[axis]
            if axis == 0:
                y0, y1 = stop - y0, stop - y1
            else:
                x0, x1 = stop - x0, stop - x1

#         import pylab as plt
#         plt.imshow(img)
#         plt.plot(x,y)
#         plt.plot((x0,x1),(y0,y1))
#         plt.show()


        return (x0, y0, x1, y1)

        

    def _findQuadLines(self):
#         img = self.img
        # TODO: give multiple options to find line
        # take first ...whatever
        #_, thresh = cv2.threshold(self._to8bitImg(self.img), 0, 255, cv2.cv.CV_THRESH_OTSU)


#         import pylab as plt
#         plt.imshow(thresh)
#         plt.show()

#         s0, s1 = img.shape
        # edge lines:

        ltop = self._findEdgeLine(self.thresh, axis=0)#, stop=s0 // 2)
        lbottom = self._findEdgeLine(self.thresh, axis=0, #start=s0 // 2,
                                     direction=-1)
        lleft = self._findEdgeLine(self.thresh, axis=1)#, stop=s1 // 2)
        lright = self._findEdgeLine(self.thresh, axis=1,# start=s1 // 2,
                                    direction=-1)
        
        return ltop, lbottom, lleft, lright

    @staticmethod
    def _verticesFromLines(l):
        ltop, lbottom, lleft, lright = l
        # grid vertices vie lines intersection:
        return np.array((ln.intersection(ltop, lleft),
                         ln.intersection(ltop, lright),
                         ln.intersection(lbottom, lright),
                         ln.intersection(lbottom, lleft)))

    @staticmethod
    def _linesFromvertices(vertices):
        c0, c1, c2, c3 = vertices
        ltop = c3[0], c3[1], c2[0], c2[1]
        lbottom = c0[0], c0[1], c1[0], c1[1]
        lleft = c0[0], c0[1], c3[0], c3[1]
        lright = c1[0], c1[1], c2[0], c2[1]
        return ltop, lbottom, lleft, lright


#     def _refineLines(self, lines, plot=True):#, plot=False, sub_height=20):
#         '''
#         fit border lines through fitting to highest gradient
#         '''
#         lines = list(lines)
#         #sign is negative, when going from low to high intensity
#         signs = (-1,1,-1,1)
#
#         sub_height = min(91,max(7,self.img.shape[0]/200))
#
#         for m, (l,sign) in enumerate(zip(lines, signs)):
#
#                 sub = alignImageAlongLine(self.img, l, sub_height)
#                 dsub = sign*cv2.Sobel(sub,cv2.CV_64F,0,1,ksize=5)
#
#                 d0, d1 = minimumLineInArray(dsub, relative=True)
#                 new_l = ln.translate2P(l,-d0, -d1)
#                 lines[m]=new_l
#
#
#                 if plot:
#                     import pylab as plt
#                     print d0,d1, l
#                     plt.figure(4)
#                     plt.imshow(self.img, interpolation='none')
#                     plt.plot((l[0],l[2]), (l[1],l[3]))
#                     plt.plot((new_l[0],new_l[2]), (new_l[1],new_l[3]), 'o-')
#
# #                     print d0,d1,l,new_l, ll[m]
#                     plt.figure(1)
#                     plt.imshow(sub, interpolation='none')
#                     plt.axes().set_aspect('auto')
#
#
#                     sub2 = alignImageAlongLine(self.img, new_l, sub_height)
#                     plt.figure(3)
#                     plt.imshow(sub2, interpolation='none')
# #                     plt.colorbar()
#                     plt.axes().set_aspect('auto')
#
#                     plt.show()
#
#
#
#         return lines


    def plot(self):
        import pylab as plt
        plt.imshow(self.img)
        for (x0,y0,x1,y1) in self._linesFromvertices(self.vertices):
            print((x0,x1), (y0,y1))
            plt.plot((x0,x1), (y0,y1))
        plt.show()


    def drawVertices(self, img=None, color=None, thickness=4):
        if img is None:
            img = self.thresh.astype(np.uint8)*128
            color = 255
        else:
            if color is None:
                color = img.max() - 1

        for l in self._linesFromvertices(self.vertices.astype(int)):
            cv2.line(img, tuple(l[:2]), tuple(l[2:]),
                     int(color), thickness=thickness)
        
            
#         img[self.thresh] += color
        return img


if __name__ == '__main__':
    import sys
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor

    p = PathStr(imgProcessor.__file__).dirname().join(
        'media', 'electroluminescence')

    img = p.join('EL_module_orig.PNG')
    q = QuadDetection(img)
    img = q.drawVertices()

    img2 = p.join('EL_cell_cracked.png')
    q = QuadDetection(img2)
    img2 = q.drawVertices(thickness=10)

    if 'no_window' not in sys.argv:
        plt.figure('module')
        plt.imshow(img)

        plt.figure('cell')
        plt.imshow(img2)

        plt.show()
