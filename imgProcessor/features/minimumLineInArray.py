import numpy as np
from numba import jit


@jit(nopython=True)   
def _lineSumXY(x,res,sub, f):
    s0 = sub.shape[0]
    sx = x.shape[0]
    hs = (s0-1)*0.5

    for i in xrange(s0):
        ff = 1 - f*(abs(i-hs)/hs)
        for j in xrange(s0):
            c = float(i)
            d = float(j-i)/sx
            val  = 0.0
            for n in xrange(sx):
                val += sub[int(round(c)),x[n]]
                c += d
            res[i,j] = val*ff
            
            
def minimumLineInArray(arr, relative=False, f=0):
    '''
    find closest minimum position next to middle line
    relative: return position relative to middle line
    f: relative decrease (0...1) - setting this value close to one will 
       discriminate positions further away from the center
    '''
    s0,s1 = arr.shape[:2]

    x = np.linspace(0,s1-1,min(100,s1), dtype=int)
    res = np.empty((s0,s0),dtype=float)

    _lineSumXY(x,res, arr,f)
#     if f != 0:
#         import pylab as plt
#         plt.imshow(res)
#         plt.colorbar()
#         plt.show()

    i,j = np.unravel_index(np.argmin(res), res.shape)

    if not relative:
        return i,j

    hs = s0/2
    dy0 = 2*(hs-i) +2#...no idea why +2
    dy1 = 2*(hs-j) +2
#     print i,j,hs
    return dy0,dy1



if __name__ == '__main__':
    import pylab as plt
    import sys
    
    arr = np.random.rand(100,200)
    s0, s1 = arr.shape
    mid = s0/2
    var = 10
    
    #draw a minimum line:
    y = np.linspace(mid-var,mid+var,s1, dtype=int)
    x = np.arange(s1)
    arr[y,x] -=1
    
    #find line:
    y0,y1 = minimumLineInArray(arr)

    yfit = (y0,y1)
    xfit = (0,s1)
    
    if 'no_window' not in sys.argv:
        #plot
        plt.figure('given array')
        plt.imshow(arr)
        
        plt.figure('found line')
        plt.imshow(arr)
        plt.plot(xfit,yfit,  linewidth=10,linestyle='--')
        
        plt.show()
