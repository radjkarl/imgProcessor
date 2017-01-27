# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
from fancytools.os.PathStr import PathStr
from imgProcessor.imgIO import imread
from imgProcessor.equations.gaussian import gaussian
# from fancytools.math.findXAt import findXAt



class FitHistogramPeaks(object):
    '''
    try to fit the histogram of an image as an addition of 2 GAUSSian distributions
    stores the position the the peaks in self.fitParams
    '''
    
    def __init__(self, img, 
                 binEveryNPxVals=10, 
                 fitFunction=gaussian,
                 maxNPeaks=4,
                 debug=False):
        '''
        :param binEveryNPxVals: how many intensities should be represented by one histogram bin
        :param fitFunction: function to fit the histogram (currently only gaussian)
        :param maxNPeaks: limit number of found peaks (biggest to smallest)
        :param debug: whether to print error messages
        
        public attributes:
        .fitParams -> list of fit parameters (for gaussian: (intensity, position, standard deviation))
        '''
        #import here to decrease startup time
        from scipy.optimize import curve_fit
        
        self.fitFunction = fitFunction
        self.fitParams = []
        ind = None
        self.img = imread(img, 'gray')
        if self.img.size > 25000:
            #img is big enough: dot need to analyse full img
            self.img = self.img[::10,::10]
        try:
            self.yvals, bin_edges = np.histogram(self.img, bins=200)
        except:
            ind = np.isfinite(self.img)
            self.yvals, bin_edges = np.histogram(self.img[ind], 
                                                 bins=200)


        self.yvals = self.yvals.astype(np.float32)
        #move histogram range to representative area: 
        cdf = np.cumsum(self.yvals) / self.yvals.sum()
#         import pylab as plt
# 
#         plt.plot(bin_edges[:-1],cdf)
#         plt.show()
        
        i0 = np.argmax(cdf>0.02)
        i1 = np.argmax(cdf>0.98)
        mnImg = bin_edges[i0]
        mxImg = bin_edges[i1]
#         print(mnImg, mxImg)
        #one bin for every  N pixelvalues
        nBins = 50#np.clip(int( ( mxImg - mnImg) / binEveryNPxVals ),25,100)
        if ind is not None:
            img = self.img[ind]

        self.yvals, bin_edges = np.histogram(img, bins=nBins, 
                                                 range=(mnImg, mxImg))
       
        #bin edges give start and end of an area-> move that to the middle:
        self.xvals = bin_edges[:-1]+np.diff(bin_edges)*0.5
        
        #in the (quite unlikely) event of two yvals being identical in sequence
        #peak detection wont work there, so remove these vals before:
        valid = np.append(np.logical_and(self.yvals[:-1]!=0, np.diff(self.yvals)!=0),True)
        self.yvals = self.yvals[valid]
        self.xvals = self.xvals[valid]
        
 
    
        
        yvals = self.yvals.copy()
        xvals = self.xvals
        s0,s1 = self.img.shape
        minY = max(10,float(s0*s1)/nBins/50)
        mindist = 5
        
        peaks = self._findPeaks(yvals,mindist, maxNPeaks, minY)

        valleys = self._findValleys(yvals, peaks)
        positions = self._sortPositions(peaks,valleys)

        #FIT FUNCTION TO EACH PEAK:
        for il,i,ir in positions:
            #peak position/value:
            xp = xvals[i]
            yp = yvals[i]
            
            xcut = xvals[il:ir]
            ycut = yvals[il:ir]

            #approximate standard deviation from FHWM:
            #ymean = 0.5* (yp + ycut[-1]) 
            #sigma = abs(xp - findXAt(xcut,ycut,ymean) )
            sigma = 0.5*abs(xvals[ir]-xvals[il])

            init_guess = (yp,xp,sigma)
            #FIT
            try:
                #fitting procedure using initial guess
                params, _ = curve_fit(self.fitFunction, xcut, ycut, 
                                      p0=init_guess,
                                      sigma=np.ones(shape=xcut.shape)*1e-8)  
            except (RuntimeError, TypeError):
                #TypeError: not enough values given (when peaks and valleys to close to each other)
                if debug:
                    print("couln't fit gaussians -> result will will inaccurate")
                #stay with initial guess: 
                params = init_guess
#             except TypeError, err:
#                 print err
#                 #couldn't fit maybe because to less values were given
#                 continue
            if (params[0]>0#has height
#                  and #has height
                #peak is within the image histogram
#                 mnImg < params[1] < mxImg
#                 (params[1]+2*params[2]> mnImg 
#                     or params[1]-2*params[2]<mxImg) 
                ):
                params = list(params)
                params[2]  = np.abs(params[2])
                
                self.fitParams.append(params)
                
            y = self.fitFunction(self.xvals,*params).astype(yvals.dtype)
            yvals -= y #peaks add up
            yvals[yvals<0]=0 #can't be negative
            
        #sort for increasing x positions
        self.fitParams = sorted(self.fitParams, key=lambda p: p[1])

#         print(self.fitParams)
#         import pylab as plt
#         plt.plot(self.xvals,self.yvals)
#         for f in self.fitValues():
#             plt.plot(self.xvals, f)
#         plt.show()

    
    @staticmethod
    def _sortPositions(peaks, valleys):
        #make a set of  from where to where to fit
        #starting with widest peak
        #also ensure that every set is at least 3 values wide
                #FIT FUNCTION TO EACH PEAK:
        positions = list(zip(valleys[:-1],peaks,valleys[1:]))
        positions = sorted(positions, key=lambda s: s[1])#s[2]-s[0])
        positions.reverse()
        #filter invalid:
        positions = [p for p in positions if p[-1]-p[0]>2]
        return positions


    @staticmethod
    def _findValleys(vals,peaks):
        assert len(peaks)>1, 'need at least 2 peaks to find valleys'
        #find minimum between peaks
        l = []
        for p0,p1 in zip(peaks[:-1],peaks[1:]): 
            l.append( p0 + max(3,np.argmin(vals[p0:p1])) ) #3 to ensure min points=3, needed for fitting
        #first valley
        l.insert(0,max(0,peaks[0]-l[0]))
        #last valley
        l.append(min(len(vals)-1, p1+p1-l[-1] ) )
        return l
        
        
    @staticmethod
    def _findPeaks(vals, mindist, maxPeaks, minVal):
        from scipy.signal import argrelextrema#save startup time
    
        #peak defined as local maximum that is not exceeded 
        #within a minimum distance
        l = len(vals)
        peaks = argrelextrema(vals, np.greater, mode='wrap')[0]
        valid = np.ones(len(peaks), dtype=bool)
        if len(peaks)>1:
            #try 4 times to filter peaks:
            for _ in range(4):
                for i,p in enumerate(peaks):
                    r0 = max(0,p-mindist)
                    r1 = min(l,p+mindist)
                    if (vals[p] < minVal or
                        vals[r0:r1].max() > vals[p] ):
                        valid[i]=False 
                if valid.sum()>1: #need at least 2 peaks
                    #only filter if peaks remain:
                    peaks = peaks[valid]
                    break
                else:
                    #reduce demand
                    mindist -= 1
                    minVal -= 1
        #add first peak at i=0 if existent
        if peaks[0] != 0 and vals[0]>vals[1] and vals[0]>minVal:
            peaks = np.insert(peaks,0,0)

        peaks = peaks[peaks.argsort()[::-1]]
        return peaks[:maxPeaks][::-1]

   
    def fitValues(self, xvals=None):
        if xvals is None:
            xvals = self.xvals
        return [self.fitFunction(xvals,a, b, c) for (a,b,c) in self.fitParams]
        


if __name__ == '__main__':
    import sys
    import imgProcessor
    from imgProcessor.scripts._FitHistogramPeaks import plotFitResult
    import pylab as plt
    imgs =  PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence').all()
    for i in imgs:
        f = FitHistogramPeaks(i)
        print(f.fitParams)
        
        if 'no_window' not in sys.argv: 
            plt.figure(1)
            plt.imshow(f.img)
            plt.colorbar()
            plotFitResult(f, save_to_file=False)