'''
functions taken from http://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry
is reference of 
http://www.sayonics.com/publications/pertuz_PR2013.pdf
Pertuz 2012: Analysis of focus measure operators for shape-from-focus

code transformed from C++.openCV -> python.cv2

RETURN: focusMeasure - parameter describing sharpness of an image
'''
from __future__ import division

import cv2
import numpy as np

def modifiedLaplacian(img):
    ''''LAPM' algorithm (Nayar89)'''
    M = np.array([-1, 2, -1])
    G = cv2.getGaussianKernel(ksize=3, sigma=-1)
    Lx = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=M, kernelY=G)
    Ly = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=G, kernelY=M)
    FM = np.abs(Lx) + np.abs(Ly)
    return cv2.mean(FM)[0]
    

def varianceOfLaplacian(img):
    ''''LAPV' algorithm (Pech2000)'''
    lap = cv2.Laplacian(img, ddepth=-1)#cv2.cv.CV_64F)
    stdev = cv2.meanStdDev(lap)[1]
    s = stdev[0]**2
    return s[0]


def tenengrad(img, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx*Gx + Gy*Gy
    mn = cv2.mean(FM)[0]
    if np.isnan(mn):
        return np.nanmean(FM)
    return mn


def normalizedGraylevelVariance(img):
    ''''GLVN' algorithm (Santos97)'''
    mean, stdev = cv2.meanStdDev(img)
    s = stdev[0]**2 / mean[0]
    return s[0]
    


if __name__ == '__main__':
    from matplotlib import pyplot
    from qtpy import QtWidgets
    import os
    import sys

    class Dialog(QtWidgets.QWidget):
        '''
        Comparison of the above described relative
        sharpness measures - on one or multiple input images
        '''
        def __init__(self, *args, **kwargs):
            QtWidgets.QWidget.__init__(self, *args, **kwargs)
            self.setWindowTitle('Sharpness comparison')
            l = QtWidgets.QHBoxLayout()
            self.setLayout(l)
            btn1 = QtWidgets.QPushButton('one file (artificial defocus)')
            btn1.clicked.connect(self.oneFile)
            l.addWidget(btn1)
            btn2 = QtWidgets.QPushButton('one folder (set of focus variations)')
            btn2.clicked.connect(self.oneFolder)
            l.addWidget(btn2)  
        def oneFile(self, evt):
            filename = QtWidgets.QFileDialog.getOpenFileName()
            if filename:
                oneFileComparison(str(filename))
        def oneFolder(self, evt):
            filename = QtWidgets.QFileDialog.getExistingDirectory()
            if filename:
                oneFolderComparison(str(filename))


    def oneFileComparison(filename):
        gaussianFilterVals = list(range(1,30,2))
        gaussianFilterVals.insert(0,0)
        img = _openImage(filename)
        fn = lambda x, img=img: img if x==0 else cv2.blur(img, (x, x) )
        _procedure(fn, gaussianFilterVals, gaussianFilterVals, 'artificial blur', filename)
            
                
    def oneFolderComparison(folder):
        d = [os.path.join(folder,x) for x in os.listdir(folder)]
        xVals = list(range(len(d)))
        _procedure(_openImage, d, xVals, 'focus variation', folder)


    def _openImage(filename):
        return cv2.imread(filename, 
                        # cv2.IMREAD_ANYDEPTH | 
                        cv2.IMREAD_GRAYSCALE)  
    
    
    def _procedure(getImgFn, args, xVals, figtitle=None, wintitle=None):

        y0,y1,y2,y3 = [],[],[],[]
        for n in range(len(xVals)):
            img = getImgFn(args[n])

            y0.append(modifiedLaplacian(img))
            y1.append(varianceOfLaplacian(img))
            y2.append(tenengrad(img))
            y3.append(normalizedGraylevelVariance(img))

        _, ax = pyplot.subplots(2,2)
        ax[0][0].plot(xVals,y0)
        ax[0][0].set_title('modifiedLaplacian')
        ax[0][1].plot(xVals,y1)
        ax[0][1].set_title('varianceOfLaplacian')
        ax[1][0].plot(xVals,y2)
        ax[1][0].set_title('tenengrad')
        ax[1][1].plot(xVals,y3)
        ax[1][1].set_title('normalizedGraylevelVariance')

        fig = pyplot.gcf()
        if wintitle is not None:
            fig.canvas.set_window_title(wintitle)
        if figtitle:
            fig.suptitle(figtitle, fontsize=14)

        #for csv export:
        editor = QtWidgets.QTextEdit()
        t = '#x, modifiedLaplacian, varianceOfLaplacian, tenengrad, normalizedGraylevelVariance\n'
        for n,x in enumerate(xVals):
            t += '%s, %s, %s, %s, %s\n' %(x, y0[n], y1[n], y2[n], y3[n])
        editor.setText(t)
        
        editor.show()  
        pyplot.show()
    
    if 'no_window' not in sys.argv: 
        app = QtWidgets.QApplication([])
        d=Dialog()
        d.show()
        sys.exit(app.exec_())    
