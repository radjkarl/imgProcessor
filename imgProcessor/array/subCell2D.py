from __future__ import division
import numpy as np



def subCell2DGenerator(arr, shape, d01=None,p01=None):
    '''
    generator to access evenly sized sub-cells in a 2d array
    returns indices and sub arrays as (i,j,sub)

    d01(tuple) - cell size in y and x


    >>> a = np.array([[[0,1],[1,2]],[[2,3],[3,4]]])
    >>> gen = subCell2DGenerator(a,(2,2))
    >>> for i,j, sub in gen: print( i,j, sub )
    0 0 [[[0 1]]]
    0 1 [[[1 2]]]
    1 0 [[[2 3]]]
    1 1 [[[3 4]]]
    '''
    for i,j,s0,s1 in subCell2DSlices(arr, shape, d01,p01):
        yield i,j, arr[s0,s1]
  
    
def rint(x):
    return int(round(x))


def subCell2DSlices(arr, shape, d01=None,p01=None):
    if p01 is not None:
        yinit,xinit = p01
    else:
        xinit,yinit=0,0
        
    x,y = xinit,yinit
    g0,g1 = shape
    s0,s1 = arr.shape[:2]
    
    if d01 is not None:
        d0,d1 = d01
    else:
        d0,d1 = s0/g0, s1/g1
        
    y1 = d0+yinit
    for i in range(g0):
        for j in range(g1):
            x1 = x+d1
            yield ( i,j, slice(max(0,rint(y)),
                               max(0,rint(y1))),
                         slice(max(0,rint(x)),
                               max(0,rint(x1))) )
            x = x1
        y = y1
        y1 = y+d0
        x = xinit


def subCell2DCoords(*args, **kwargs):
    for _,_,s0,s1 in subCell2DSlices(*args, **kwargs):    
        yield ( (s1.start, s1.start, s1.stop),
                (s0.start, s0.stop, s0.stop) )


def subCell2DFnArray(arr, fn, shape, dtype=None, **kwargs):
    '''
    Return array where every cell is the output of a given cell function
    mx = subCell2DFnArray(myArray, np.max, (10,6) )
    --> here mx is a 2d array containing all cell maxima
    '''
    out = np.empty(shape, dtype=dtype)
    for i,j,c in subCell2DGenerator(arr, shape, **kwargs):
        out[i,j] = fn(c)
    return out



if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
    import pylab as plt
    import sys
    
    r0,r1 = 100,100
    shape = 8,10
    arr = np.fromfunction(lambda x,y: np.sin(x/r0)+np.cos(np.pi*y/r1), (r0,r1))
    
    avg0 = subCell2DFnArray(arr, np.mean, shape)
    gen0 = subCell2DCoords(arr, shape)
    
    p01=(20,15) 
    d01=(5,4)
    gen1 = subCell2DCoords(arr, shape, p01=p01, d01=d01)
    avg1 = subCell2DFnArray(arr, np.mean, shape, p01=p01, d01=d01)
    
    p01=(-5,-7) 
    d01=(12,20)
    gen2 = subCell2DCoords(arr, shape, p01=p01, d01=d01)
    avg2 = subCell2DFnArray(arr, np.mean, shape, p01=p01, d01=d01)

    if 'no_window' not in sys.argv:
        #PLOT:
        f, ax = plt.subplots(2,3)
        ax[0,0].set_title('{} grid in top of test image'.format(shape))
        ax[0,0].imshow(arr, clim=(-2,2))
        for x,y in gen0:
            ax[0,0].plot(x,y)
        ax[1,0].set_title('Grid average')
        ax[1,0].imshow(avg0, clim=(-2,2), interpolation='none')
        
        ax[0,1].set_title('Grid with offset and fixed width')
        ax[0,1].imshow(arr, clim=(-2,2))
        for x,y in gen1:
            ax[0,1].plot(x,y)
        ax[1,1].set_title('Grid average')
        ax[1,1].imshow(avg1, 
                        clim=(-2,2), interpolation='none')
    
        ax[0,2].set_title('Grid with negative offset')
        ax[0,2].imshow(arr, clim=(-2,2))
        for x,y in gen2:
            ax[0,2].plot(x,y)
        ax[1,2].set_title('Grid average')
        ax[1,2].imshow(avg2, clim=(-2,2), interpolation='none')
    
        plt.show()
    
    

