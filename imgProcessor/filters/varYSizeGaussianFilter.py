import numpy as np
from numba import jit
from scipy.ndimage.filters import gaussian_filter

from imgProcessor.filters._extendArrayForConvolution import extendArrayForConvolution
   
      
          
def varYSizeGaussianFilter(arr, stdyrange, stdx=0,
                           modex='wrap', modey='reflect'):
    '''
    applies gaussian_filter on input array
    but allowing variable ksize in y
    
    stdyrange(int) -> maximum ksize - ksizes will increase from 0 to given value
    stdyrange(tuple,list) -> minimum and maximum size as (mn,mx)
    stdyrange(np.array) -> all different ksizes in y
    '''
    assert arr.ndim == 2, 'only works on 2d arrays at the moment'
    
    s0 = arr.shape[0]
    
    #create stdys:
    if isinstance(stdyrange, np.ndarray):
        assert len(stdyrange)==s0, '[stdyrange] needs to have same length as [arr]'
        stdys = stdyrange
    else:
        if type(stdyrange) not in (list, tuple):
            stdyrange = (0,stdyrange)
        mn,mx = stdyrange
        stdys  = np.linspace(mn,mx,s0)
    
    #prepare array for convolution:
    kx = int(stdx*2.5)
    kx += 1-kx%2
    ky = int(mx*2.5)
    ky += 1-ky%2
    arr2 = extendArrayForConvolution(arr, (kx, ky), modex, modey)
    
    #create convolution kernels:
    inp = np.zeros((ky,kx))
    inp[ky//2, kx//2] = 1
    kernels = np.empty((s0,ky,kx))
    for i in range(s0):
        stdy = stdys[i]
        kernels[i] = gaussian_filter(inp, (stdy,stdx))

    out = np.empty_like(arr)
    _2dConvolutionYdependentKernel(arr2, out, kernels)
    return out


@jit(nopython=True)
def _2dConvolutionYdependentKernel(arr, out, kernels):
    s0,s1 = arr.shape
    k0,k1 = kernels.shape[1:]
    hk0 = k0//2
    hk1 = k1//2
    for i in range(k0,s0-k0):
        for j in  range(k1,s1-k1):
            #2d convolution:
            v = 0
            for ii in range(k0):
                for jj in  range(k1):
                    a = arr[i+ii-hk0, j+jj-hk1]
                    if not np.isnan(a):
                        v += kernels[i-k0,ii,jj] * a
            out[i-k0,j-k1] = v



if __name__ == '__main__':
    import cv2
    import pylab as plt
    import sys
    
    res = (50,50)
    rad = 20
    stdy = (0,10)

    arr = np.zeros(res)
    cv2.circle(arr, (res[1]//2, rad), rad, 1, -1)

    arr2 = varYSizeGaussianFilter(arr, stdy,0)
    
    if 'no_window' not in sys.argv:
        f, (arr0,arr1) = plt.subplots(2)
        arr0.set_title('input')
        arr0.imshow(arr, interpolation='none')
        arr1.set_title('output with ksize in y from %s to %s' %(stdy[0], stdy[1]))
        arr1.imshow(arr2, interpolation='none')
        plt.show()
    
    