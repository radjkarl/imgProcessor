import cv2
import numpy as np

from scipy.signal import convolve2d


def _mkConvKernel(ksize, orientations, image):
    assert ksize[0]%2 and ksize[1]%2
    dangle = np.pi/orientations
    angle = 0
    k0,k1 = ksize
    mx,my = k0/2+1,k1/2+1
#     length = 0.5*(k0+k1)

    kernel = np.empty( (orientations,k0,k1) )
    for i in xrange(orientations):
        #make line kernel
        x = int(round(4*np.cos(angle)*k0))
        y = int(round(4*np.sin(angle)*k1))
        k = np.zeros((2*k0,2*k1), dtype=np.uint8)
        cv2.line(k, (-x+k0,-y+k1), (x+k0,y+k1), 
                 255, 
                 thickness=1, lineType=cv2.CV_AA)
        #resize and scale 0-1:
        ki = k[mx:mx+k0,my:my+k1].astype(float)/255
        kernel[i] = ki / ki.sum()
#         ki /= ki.sum()
        angle+=dangle
#     kernel /= kernel.sum(axis=(1,2))
#     kernel/=length
    return kernel
        

def hog(image, orientations=6, ksize=(5,5)):
    '''
    returns the Histogram of Oriented Gradients
    
    :param ksize: convolution kernel size as (y,x) - needs to be odd 
    :param orientations: number of orientations in between rad=0 and rad=pi
    
    similar to http://scikit-image.org/docs/dev/auto_examples/plot_hog.html
    but faster and with less options
    '''
    s0,s1 = image.shape

    k = _mkConvKernel(ksize, orientations, image)
    out = np.empty(shape=(s0,s1,orientations))
    for i in xrange(orientations):
        out[:,:,i] = convolve2d(image, k[i], mode='same' )          
    return out


if __name__ == '__main__':
    import sys
    from imgProcessor.imgIO import imread
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    
    p = PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence')
    
    img = imread(p.join('EL_cell_cracked.PNG'), 'gray')

    k = (15,15)
    o = 4
    p0=(141,91)
    p1=(251,242)
    p2=(667,570)

    kernels = _mkConvKernel(k, o, img)
    h = hog(img, ksize=k, orientations=o)

    if 'no_window' not in sys.argv:
        plt.figure('image')
        plt.imshow(img)
        plt.scatter(*p0)
        plt.scatter(*p1)
        plt.scatter(*p2)
        
        plt.figure('orientations')
        plt.imshow(kernels.sum(axis=0), interpolation='none')
    
        plt.figure('HOG at selected positions')
        plt.plot(h[p0])
        plt.plot(h[p1])
        plt.plot(h[p2])
        
        plt.show()
