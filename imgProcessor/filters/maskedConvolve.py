'''
Created on 30 Dec 2016

@author: elkb4
'''
import numpy as np
from numba import njit
from imgProcessor.filters._extendArrayForConvolution import extendArrayForConvolution




def maskedConvolve(arr, kernel, mask, mode='reflect'):
    '''
    same as scipy.ndimage.convolve but is only executed on mask==True
    ... which should speed up everything
    '''
    arr2 = extendArrayForConvolution(arr, kernel.shape, modex=mode, modey=mode)
    print(arr2.shape)
    out = np.zeros_like(arr)
    return _calc(arr2, kernel, mask, out)


@njit
def _calc(arr, kernel, mask, out):
    gx = arr.shape[0]
    gy = arr.shape[1]
    kx,ky = kernel.shape
    hkx = kx//2
    hky = ky//2
    for i in range(hkx,gx-hkx):
        for j in range(hky,gy-hky):   
            #print (mask[i,j])
            if mask[i-hkx,j-hky]:

                val = 0

                for ii in range(-hkx,hkx+1):
                    for jj in range(-hky,hky+1):
                        val += kernel[ii,jj]*arr[i+ii,j+jj]

                    out[i-hkx,j-hky] = val
    return out



if __name__ == '__main__':
    import sys
    import pylab as plt
    from timeit import default_timer

    from scipy.ndimage import convolve


    
    arr = np.fromfunction(lambda x,y:np.sin(x)+np.cos(y), (3000,4000))
    k = np.eye(5)
    mask = np.zeros_like(arr, dtype=bool)
    mask[2000:2010]=True
    mask[:,2000:2010]=True
    

    start = default_timer()
    out = maskedConvolve(arr, k, mask)
    stop = default_timer()

    start2 = default_timer()    
    out2 = convolve(arr, k)
    out2[~mask]=0
    stop2 = default_timer()
    
    #prove that both methods are the same:
    assert(np.allclose(out,out2))
    
    #show that method should be faster:
    print('execution time this method: %s - exec.time scipy.method: %s' %(stop-start, stop2-start2))
    #however .... is isnt. and here i stop.
    
    if 'no_window' not in sys.argv:
        plt.imshow(out)
        plt.colorbar()
        
        plt.figure(2)
        plt.imshow(out2)
        plt.colorbar()
        plt.show()
    
    