import numpy as np
from numba import jit, boolean



def interpolate2dStructuredPointSpreadIDW(grid, mask, kernel=15, power=2, maxIter=1e5, copy=True):
    '''
    same as interpolate2dStructuredIDW but using the point spread method
    this is faster if there are bigger connected masked areas and the border length is smaller
    
    replace all values in [grid] indicated by [mask]
    with the inverse distance weighted interpolation of all values within 
    px+-kernel
    
    [power] -> distance weighting factor: 1/distance**[power]
    [copy] -> False: a bit faster, but modifies 'grid' and 'mask'
    '''
    assert grid.shape==mask.shape, 'grid and mask shape are different'
    
    border = np.zeros(shape=mask.shape, dtype=np.bool)
    if copy:
        #copy mask as well because if will be modified later:
        mask = mask.copy()
        grid = grid.copy()
    return _calc(grid, mask, border, kernel, power, maxIter)
    


@jit(boolean(boolean[:,:], boolean[:,:]), nopython=True)
def _createBorder(mask, border):
    #create a 1px-width-border around all mask=True values

    gx = mask.shape[0]
    gy = mask.shape[1]
    last_val = mask[0,0]
    val = last_val
    any_border = False
    for i in xrange(gx):
        for j in xrange(gy):
            val = mask[i,j]
            if val != last_val:
                if val:
                    border[i,j] = True
                else:
                    border[i,j-1] = True
                any_border = True 
            last_val = val
    last_val = mask[0,0]
    for j in xrange(gy):
        for i in xrange(gx):
            val = mask[i,j]
            if val != last_val:
                if val:
                    border[i,j] = True
                else:
                    border[i-1,j] = True
                any_border = True 
 
            last_val = val
    return any_border


@jit(nopython=True)
def _calc(grid, mask, border, kernel, power, maxIter):

    any_border = _createBorder(mask, border)

    gx = mask.shape[0]
    gy = mask.shape[1]

    n = 0
    while n< maxIter and any_border:
        for i in xrange(gx):
            for j in xrange(gy):
                if border[i,j]:
                              
                    xmn = i-kernel
                    if xmn < 0:
                        xmn = 0
                    xmx = i+kernel
                    if xmx > gx:
                        xmx = gx
                          
                    ymn = j-kernel
                    if ymn < 0:
                        ymn = 0
                    ymx = j+kernel
                    if ymx > gx:
                        ymx = gy
                          
                    sumWi = 0.0
                    value = 0.0 
                                          
                    for xi in xrange(xmn,xmx):
                        for yi in xrange(ymn,ymx):
                            if  not (xi == i and yi == j) and not mask[xi,yi]:
                                wi = 1.0 / ((xi-i)**2+(yi-j)**2)**(0.5*power )
                                sumWi += wi
                                value += wi * grid[xi,yi]  
                    if sumWi:
                        #print value
                        grid[i,j] = value/sumWi               
      
                        border[i,j] = False
                        mask[i,j] = False
        
        any_border = _createBorder(mask, border)
        n += 1

    return grid



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    shape = (1000,1000)

    arr = np.random.rand(*shape)
    mask = np.random.randint(100, size=shape)
    mask = (mask>5).astype(bool)

    arr2 = interpolate2dStructuredPointSpreadIDW(arr, mask, 5,4)


    if 'no_window' not in sys.argv:
        arr[mask]=np.nan
        plt.figure('original data')
        plt.imshow(arr, interpolation='none')
        plt.colorbar()        
        
        plt.figure('interpolation of sparse array using point spead algorithm')
        plt.imshow(arr2, interpolation='none')
        plt.colorbar()
    
        plt.figure('used mask')
        plt.imshow(mask, interpolation='none')
        plt.colorbar()
        plt.show()
        