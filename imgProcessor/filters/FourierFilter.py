# -*- coding: utf-8 -*-
import numpy as np

from imgProcessor.imgIO import imread



class FourierFilter(object):
    '''
    http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

    Tsai, D.-M., Wu, S.-C., & Li, W.-C. (2012). Defect detection of solar cells in electroluminescence images using Fourier image reconstruction. Solar Energy Materials and Solar Cells, 99, 250�262. doi:10.1016/j.solmat.2011.12.007
    '''

    def __init__(self, img):
        self.img = imread(img)
        # Fourier transform giving a complex array
        # with zero frequency component (DC component) will be at top left corner
        self.fourier = np.fft.fft2(self.img)
        self.fshift = np.fft.fftshift(self.fourier)
            

    def highPassFilter(self, threshold):
        '''
        remove all low frequencies by setting a square in the middle of the 
        Fourier transformation of the size (2*threshold)^2 to zero
        threshold = 0...1
        '''
        if not threshold:
            return
        rows, cols = self.img.shape
        tx = int(cols*threshold)
        ty = int(rows*threshold)
        #middle:
        crow,ccol = rows/2 , cols/2
        # square in the middle to zero
        self.fshift[crow-tx:crow+tx, ccol-ty:ccol+ty] = 0


    def lowPassFilter(self, threshold):
        '''
        remove all high frequencies by setting boundary around a quarry in the middle
        of the size (2*threshold)^2 to zero
        threshold = 0...1
        '''
        if not threshold:
            return
        rows, cols = self.img.shape
        tx = int(cols*threshold*0.25)
        ty = int(rows*threshold*0.25)
        #upper side
        self.fshift[rows-tx:rows, : ] = 0
        #lower side
        self.fshift[0:tx, : ] = 0
        #left side
        self.fshift[: , 0:ty ] = 0
        #right side
        self.fshift[: , cols-ty:cols ] = 0     


    def suppressValuesSmallerThanThresholdToZero(self, threshold):
        if not threshold:
            return   
        with np.errstate(divide='ignore'):
            magSpectrum = np.log(np.abs(self.fshift))
        threshold *= np.nanmax(magSpectrum)
        self.fshift[magSpectrum < threshold] = 0


    def suppressValuesBiggerThanThresholdToZero(self, threshold):
        if not threshold:
            return   
        magSpectrum = np.log(np.abs(self.fshift))
        threshold = (1- threshold) * np.nanmax(magSpectrum)
        self.fshift[magSpectrum > threshold] = 0


    def deleteArea(self, fromx, fromy, tox, toy):
        if fromx==tox or fromy==toy: #need to have a rectangle
            return
        #sort
        fromx, tox = min(fromx, tox), max(fromx, tox)
        fromy, toy = min(fromy, toy), max(fromy, toy)
        #make absolute
        rows, cols = self.fshift.shape
        fromx = int(fromx*rows/2)
        tox = int(tox*rows/2)
        fromy = int(fromy*cols/2)
        toy = int(toy*cols/2)
        #upper left
        self.fshift[fromx:tox, fromy:toy] = 0
        #upper right
        self.fshift[rows-tox:rows-fromx, fromy:toy] = 0
        #lower left
        self.fshift[fromx:tox, cols-toy:cols-fromy] = 0
        #lower right
        self.fshift[rows-tox:rows-fromx, cols-toy:cols-fromy] = 0


    def magnitudeSpectrum(self):
        with np.errstate(divide='ignore'):
            #ignore runtimeWarning because of value/0
            magSpectrum = np.log(np.abs(self.fshift))
            #get rid of all inf
        magSpectrum[magSpectrum == np.inf] = np.nanmax(magSpectrum)
        magSpectrum[magSpectrum == -np.inf] = np.nanmax(magSpectrum)
        return magSpectrum
    
    
    def reconstructImage(self):
        '''
        do inverse Fourier transform and return result
        '''
        f_ishift = np.fft.ifftshift(self.fshift)
        return np.real( np.fft.ifft2(f_ishift) )