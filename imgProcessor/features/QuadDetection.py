from __future__ import division

import numpy as np

import cv2

from scipy.ndimage.filters import maximum_filter, minimum_filter

from fancytools.math import line as ln
from fancytools.fit.polyFitIgnoringOutliers import polyFitIgnoringOutliers

from imgProcessor.camera.PerspectiveCorrection import PerspectiveCorrection
from imgProcessor.transformations import toUIntArray
from imgProcessor.imgSignal import signalRange, signalMinimum
from imgProcessor.imgIO import imread


class QuadDetection(object):
    '''
    detect the corners of a (bright) quadrilateral object in an image
    e.g. a PV cell/module in an EL image
    '''

    def __init__(self, img=None  # , vertices=None#, refinePositions=True
                 ):
        '''
        @param img -> input image
        @paramn vertices -> routh estimate of corner positions
        @refinePositions -> whether to refine (found) corner positions
        '''
        self.img = imread(img, 'gray')
#         self.vertices = vertices

        self._pc = None

#         if self.vertices is None:
        lines = self._findQuadLines()

#             if refinePositions:
#                 lines = self._refineLines(lines)
        self.vertices = self._verticesFromLines(lines)

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
    def _findEdgeLine(img, axis=0, start=0, stop=-1, direction=1):
        # find the approximate edge line of and object within an image
        # along a given axis
        s = img.shape
        if start != 0 or stop != -1:
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

#         ind = np.argmax(img, axis=axis)
#         arr = np.zeros(img.shape)
#         try:
#             arr[x, ind]=-1
#         except IndexError:
#             arr[ind,x]=-1
#
#         import pylab as plt
#         plt.imshow(arr)
#         plt.show()

        # return first non=zero value along given axis
        y = np.argmax(img, axis=axis)
        valid = y != 0
        if valid.sum() > 0.2 * sx:
            y = y[valid]
            x = x[valid]


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
        p = polyFitIgnoringOutliers(x, y, deg=1, niter=5, nstd=1)

#         import pylab as plt
#         plt.plot(x,y)
#         plt.plot((x0,x1),(y0,y1))
#         plt.show()

        # extract edge points:
        x0, x1 = x[0], x[-1]
        y0, y1 = p(x0), p(x1)
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
            if stop == -1:
                stop = s[axis]
            if axis == 0:
                y0, y1 = stop - y0, stop - y1
            else:
                x0, x1 = stop - x0, stop - x1

        return (x0, y0, x1, y1)

    def _findQuadLines(self):
        img = self.img
        # TODO: give multiple options to find line
        # take first ...whatever
        #_, thresh = cv2.threshold(self._to8bitImg(self.img), 0, 255, cv2.cv.CV_THRESH_OTSU)
        thresh = img > signalMinimum(img)

        # remove small features:
        thresh = minimum_filter(thresh, 5)
        thresh = maximum_filter(thresh, 5)

#         import pylab as plt
#         plt.imshow(thresh)
#         plt.show()

        s0, s1 = img.shape
        # edge lines:
        ltop = self._findEdgeLine(thresh, axis=0, stop=s0 // 2)
        lbottom = self._findEdgeLine(thresh, axis=0, start=s0 // 2,
                                     direction=-1)
        lleft = self._findEdgeLine(thresh, axis=1, stop=s1 // 2)
        lright = self._findEdgeLine(thresh, axis=1, start=s1 // 2,
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

    def drawVertices(self, img=None, color=None, thickness=4):
        if img is None:
            img = self.img
        if color is None:
            color = img.max() - 1
        for l in self._linesFromvertices(self.vertices.astype(int)):
            cv2.line(img, tuple(l[:2]), tuple(l[2:]),
                     int(color), thickness=thickness)
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
