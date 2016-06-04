# -*- coding: utf-8 -*-

import numpy as np
from fancytools.os.PathStr import PathStr
from imgProcessor.imgIO import imread
from imgProcessor.equations.gaussian import gaussian



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
        self.img = imread(img, 'gray')
        if self.img.size > 25000:
            #img is big enough: dot need to analyse full img
            self.img = self.img[::10,::10]

        self.yvals, bin_edges = np.histogram(self.img, bins=200)
        self.yvals = self.yvals.astype(np.float32)
        #move histogram range to representative area: 
        cdf = np.cumsum(self.yvals)/self.yvals.sum()
        i0 = np.argmax(cdf>0.02)
        i1 = np.argmax(cdf>0.98)
        mnImg = bin_edges[i0]
        mxImg = bin_edges[i1]
        #one bin for every  N pixelvalues
        nBins = np.clip(int( ( mxImg - mnImg)  / binEveryNPxVals ),25,50)
        self.yvals, bin_edges = np.histogram(self.img, bins=nBins, 
                                                 range=(mnImg, mxImg))
        #bin edges give start and end of an area-> move that to the middle:
        self.xvals = bin_edges[:-1]+np.diff(bin_edges)*0.5
        
        self.fitParams = []
        
        yvals = self.yvals.copy()
        xvals = self.xvals
        s0,s1 = self.img.shape
        minY = max(10,float(s0*s1)/nBins/100)
        
        peaks = self._findPeaks(yvals,5, maxNPeaks, minY)
        valleys = self._findValleys(yvals, peaks)
        positions = self._sortPositions(peaks,valleys)

        #FIT FUNCTION TO EACH PEAK:
        for il,i,ir in positions:
            #peak position/value:
            xp = xvals[i]
            yp = yvals[i]
            
            sigma = 0.5*abs(xvals[ir]-xvals[il])
            xcut = xvals[il:ir]
            ycut = yvals[il:ir]
            init_guess = (yp,xp,sigma)
            #FIT
            try:
                #fitting procedure using initial guess
                params, _ = curve_fit(self.fitFunction, xcut, ycut, 
                                      p0=init_guess,
                                      sigma=np.ones(shape=xcut.shape)*1e-8)  
            except RuntimeError:
                if debug:
                    print "couln't fit gaussians -> result will will inaccurate"
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


    @staticmethod
    def _sortPositions(peaks, valleys):
        #make a set of  from where to where to fit
        #starting with widest peak
        #also ensure that every set is at least 3 values wide
                #FIT FUNCTION TO EACH PEAK:
        positions = zip(valleys[:-1],peaks,valleys[1:])
        positions = sorted(positions, key=lambda s: s[2]-s[0])
        positions.reverse()
        return positions


    @staticmethod
    def _findValleys(vals,peaks):
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
        for i,p in enumerate(peaks):
            r0 = max(0,p-mindist)
            r1 = min(l,p+mindist)
            if (vals[p] < minVal or
                vals[r0:r1].max() > vals[p] ):
                valid[i]=False 
        peaks = peaks[valid]

        #add first peak at i=0 if existent
        if peaks[0] != 0 and vals[0]>vals[1] and vals[0]>minVal:
            peaks = np.insert(peaks,0,0)

        peaks = peaks[peaks.argsort()[::-1]]
        return peaks[:maxPeaks][::-1]

   
    def fitValues(self, xvals=None):
        if xvals is None:
            xvals = self.xvals
        return [self.fitFunction(xvals,a, b, c) for (a,b,c) in self.fitParams]
        
        
        
def plotFitResult(fit, show_legend=True, show_plots=True, save_to_file=False, foldername='', filename='', filetype='png'):
    from matplotlib import pyplot

    xvals = fit.xvals
    yvals = fit.yvals
    
    fit  = fit.fitValues(xvals)

    fig, ax = pyplot.subplots(1)

    ax.plot(xvals, yvals, label='histogram', linewidth=3)

    for n,f in enumerate(fit):
        ax.plot(xvals, f, label='peak %i' %(n+1), linewidth=6)

    l2 = ax.legend(loc='upper center', bbox_to_anchor=(0.7, 1.05),
      ncol=3, fancybox=True, shadow=True)
    l2.set_visible(show_legend)
    
    pyplot.xlabel('pixel value')
    pyplot.ylabel('number of pixels')
    
    if save_to_file:
        p = PathStr(foldername).join(filename).setFiletype(filetype)
        pyplot.savefig(p)
        with open(PathStr(foldername).join('%s_params.csv' %filename), 'w') as f:
            f.write('#x, #y, #fit\n')
            for n, (x,y,ys) in enumerate(zip(xvals,yvals)):
                fstr = ', '.join(str(f[n]) for f in fit)
                f.write('%s, %s, %s\n' %(x,y,fstr))
        
    if show_plots:
        pyplot.show()


#REMOVE? or into scripts
def plotSet(imgDir, posExTime, outDir, show_legend, show_plots, save_to_file, ftype):
    '''
    creates plots showing both found GAUSSIAN peaks, the histogram, a smoothed histogram 
    from all images within [imgDir] 
    
    posExTime - position range of the exposure time in the image name e.g.: img_30s.jpg -> (4,5)
    outDir - dirname to save the output images
    show_legend - True/False
    show_plots - display the result on screen
    save_to_file - save the result to file
    ftype - file type of the output images
    '''
    from matplotlib import pyplot

    xvals = []
    hist = []
    peaks = []
    exTimes = []
    max_border = 0

    if not imgDir.exists():
        raise Exception("image dir doesn't exist")

    for n,f in enumerate(imgDir):
        print f
        try:
        #if imgDir.join(f).isfile():
            img = imgDir.join(f)
            s = FitHistogramPeaks(img)
            xvals.append(s.xvals)
            hist.append(s.yvals)
#             smoothedHist.append(s.yvals2)
            peaks.append(s.fitValues())
            
            if s.border() > max_border:
                max_border = s.plotBorder()
                
            exTimes.append(float(f[posExTime[0]:posExTime[1]+1]))
        except:
            pass
    nx = 2
    ny = int(len(hist)/nx) + len(hist) % nx

    fig, ax = pyplot.subplots(ny,nx)
    
    #flatten 2d-ax list:
    if nx > 1:
        ax = [list(i) for i in zip(*ax)] #transpose 2d-list
        axx = []
        for xa in ax:
            for ya in xa:
                axx.append(ya)
        ax = axx
    
    for x,h,p,e, a in zip(xvals, hist,peaks, exTimes, ax):

        a.plot(x, h, label='histogram', thickness=3)
#         l1 = a.plot(x, s, label='smoothed')
        for n,pi in enumerate(p):
            l2 = a.plot(x, pi, label='peak %s' %n, thickness=6)
        a.set_xlim(xmin=0, xmax=max_border)
        a.set_title('%s s' %e)
        
#         pyplot.setp([l1,l2], linewidth=2)#, linestyle='--', color='r')       # set both to dashed

 
    l1 = ax[0].legend()#loc='upper center', bbox_to_anchor=(0.7, 1.05),
    l1.draw_frame(False)


    pyplot.xlabel('pixel value')
    pyplot.ylabel('number of pixels')
    
    fig = pyplot.gcf()
    fig.set_size_inches(7*nx, 3*ny)
    
    if save_to_file:
        p = PathStr(outDir).join('result').setFiletype(ftype)
        pyplot.savefig(p, bbox_inches='tight')
        
    if show_plots:
        pyplot.show()




if __name__ == '__main__':
    import sys
    import imgProcessor
    import pylab as plt
    imgs =  PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence').all()

    for i in imgs:
        f = FitHistogramPeaks(i)
        print f.fitParams
        
        if 'no_window' not in sys.argv: 
            plt.figure(1)
            plt.imshow(f.img)
            plt.colorbar()
            plotFitResult(f, save_to_file=False)