import numpy as np
from scipy.ndimage.filters import gaussian_filter
from numba import jit



def standardDeviation2d(img, ksize=5, blurred=None):
    '''
    calculate the spatial resolved standard deviation 
    for a given 2d array
    
    ksize   -> kernel size
    
    blurred(optional) -> with same ksize gaussian filtered image
                      setting this parameter reduces processing time
    '''
    if ksize not in (list, tuple):
        ksize = (ksize,ksize)

    if blurred is None:
        blurred = gaussian_filter(img, ksize)
    else:
        assert blurred.shape == img.shape
    
    std = np.empty_like(img)
    
    _calc(img, ksize[0], ksize[1], blurred, std)
    
    return std


@jit(nopython=True)
def _calc(img, ksizeX, ksizeY, blurred, std):

    gx = img.shape[0]
    gy = img.shape[1]

    npx = ksizeX*ksizeY
    
    hkx = ksizeX//2
    hky = ksizeY//2

    for i in xrange(gx):
        for j in xrange(gy):
            #get kernel boundaries:
            xmn = i-hkx
            if xmn < 0:
                xmn = 0
            xmx = i+hkx
            if xmx > gx:
                xmx = gx
                
            ymn = j-hky
            if ymn < 0:
                ymn = 0
            ymx = j+hky
            if ymx > gy:
                ymx = gy
            
            val = 0
            mean = blurred[i,j]
            #calculate local standard deviation:            
            for ii in xrange(xmx-xmn):
                for jj in xrange(ymx-ymn):
                    val += ( img[xmn+ii,ymn+jj] - mean )**2

            npx = ii*jj    
            std[i,j] = (val/npx)**0.5 



if __name__ == '__main__':
    import sys
    from matplotlib import pyplot as plt
    
    img = np.random.rand(100,100)
    std = standardDeviation2d(img, ksize=11)
    
    if 'no_window' not in sys.argv:
        plt.figure('input')
        plt.imshow(img)
        
        plt.figure('standard deviation')
        plt.imshow(std)
        
        plt.show()
    
