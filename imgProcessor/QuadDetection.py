import numpy as np

import cv2

from scipy.ndimage.filters import maximum_filter,minimum_filter

from fancytools.math import line as ln
from fancytools.fit.polyFitIgnoringOutliers import polyFitIgnoringOutliers

from imgProcessor.cameraCalibration.PerspectiveCorrection import PerspectiveCorrection
from imgProcessor.transformations import toUIntArray
from imgProcessor.alignImageAlongLine import alignImageAlongLine
from imgProcessor.scaleSignal import signalRange
from imgProcessor.minimumLineInArray import minimumLineInArray
from imgProcessor.imgIO import imread
from imgProcessor.scaleSignal import signalMinimum


class QuadDetection(object):
    '''
    detect the corners of a (bright) quadrilateral object in an image
    '''
    
    def __init__(self, img=None, vertices=None, refinePositions=True):#, object_is_bright=True):
        '''
        @param img -> input image
        @paramn vertices -> routh estimate of corner positions
        @refinePositions -> whether to refine (found) corner positions
        '''
        self.img = imread(img, 'gray')
        self.vertices = vertices
        
        self._pc = None
        
        if self.vertices is None:
            lines = self._findQuadLines()
            if refinePositions:
                lines = self._refineLines(lines)
            self.vertices = self._verticesFromLines(lines)
      

    def correct(self, img=None,**kwargs): 
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


    @staticmethod
    def _to8bitImg(img):
        if img.dtype == np.uint8:
            return img
        r = signalRange(img)
        return toUIntArray(img, dtype=np.uint8, range=r)


    @staticmethod
    def _findEdgeLine(img, axis=0, start=0, stop=-1, direction=1):
        #find the approximate edge line of and object within an image
        #along a given axis
        s = img.shape
        if start != 0 or stop != -1:
            #cut image
            if direction == 1:
                cut = slice(start,stop)
            else:
                cut = slice(stop,start,-1)
            if axis==0:
                img = img[cut,:]
            else:
                img = img[:,cut]
        sx = s[int(~axis)]
        x = np.arange(sx)
        #return first non=zero value along given axis
        y = np.argmax(img, axis=axis)
        valid = y!=0
        if valid.sum() > 0.2*sx:
            y = y[valid]
            x = x[valid]
        #filter outliers:
        p = polyFitIgnoringOutliers(x,y,deg=1, niter=5, nstd=1)

        #extract edge points:
        x0,x1 = x[0],x[-1]
        y0,y1 = p(x0), p(x1)
        if axis == 1:
            x0,x1,y0,y1 = y0,y1,x0,x1 
        #move points to actual position in image
        if direction == 1:
            if start != 0:
                if axis == 0:
                    y0, y1 = y0+start, y1+start
                else:
                    x0, x1 = x0+start, x1+start
        else:
            if stop == -1:
                stop = s[axis]
            if axis == 0:
                y0, y1 = stop-y0, stop-y1
            else:
                x0, x1 = stop-x0, stop-x1
  
        return (x0,y0,x1,y1)


    def _findQuadLines(self):
        img = self.img
        #TODO: give multiple options to find line 
            #take first ...whatever
        #_, thresh = cv2.threshold(self._to8bitImg(self.img), 0, 255, cv2.cv.CV_THRESH_OTSU)
        thresh = img > signalMinimum(img)
        #remove small features:
        thresh = minimum_filter(thresh,5)
        thresh = maximum_filter(thresh,5)

        s0,s1 = img.shape
        #edge lines:
        ltop = self._findEdgeLine(   thresh, axis=0, stop=s0/2 )
        lbottom = self._findEdgeLine(thresh, axis=0, start=s0/2, direction=-1)
        lleft = self._findEdgeLine(  thresh, axis=1, stop=s1/2 )
        lright = self._findEdgeLine( thresh, axis=1, start=s1/2, direction=-1)
        return ltop, lbottom, lleft, lright
        
        
    @staticmethod
    def _verticesFromLines((ltop, lbottom, lleft, lright)):
        #grid vertices vie lines intersection:
        return np.array((ln.intersection(ltop, lleft),  
                         ln.intersection(ltop, lright),
                         ln.intersection(lbottom, lright),
                         ln.intersection(lbottom, lleft) ) )
        
        
    @staticmethod
    def _linesFromvertices(vertices):
        c0,c1,c2,c3 = vertices
        ltop = c3[0],c3[1],c2[0],c2[1]
        lbottom = c0[0],c0[1],c1[0],c1[1]
        lleft = c0[0],c0[1],c3[0],c3[1]
        lright = c1[0],c1[1],c2[0],c2[1]
        return ltop, lbottom, lleft, lright
        

    def _refineLines(self, lines):#, plot=False, sub_height=20):
        '''
        fit border lines through fitting to highest gradient
        '''
        lines = list(lines)
        #sign is negative, when going from low to high intensity
        signs = (1,-1,1,-1)

        sub_height = min(91,max(7,self.img.shape[0]/200))
         
        for m, (l,sign) in enumerate(zip(lines, signs)):

                sub = alignImageAlongLine(self.img, l, sub_height)
                dsub = sign*cv2.Sobel(sub,cv2.CV_64F,0,1,ksize=5)

                d0, d1 = minimumLineInArray(dsub, relative=True)
                new_l = ln.translate2P(l,d0, d1)
                lines[m]=new_l
        return lines
                    

    def drawVertices(self, img=None, color=None, thickness=4):
        if img is None:
            img = self.img
        if color is None:
            color = img.max()-1
        for l in self._linesFromvertices(self.vertices.astype(int)):
            cv2.line(img, tuple(l[:2]), tuple(l[2:]), color, thickness=thickness)  
        return img



if __name__ == '__main__':
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor

    img = PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence', 'EL_module_orig.PNG')

    q = QuadDetection(img)
    img = q.drawVertices()

    plt.imshow(img)
    plt.show()