import numpy as np
from numba import jit


@jit(nopython=True)   
def _lineSumXY(x,res,sub):
    s0 = sub.shape[0]
    sx = x.shape[0]

    for i in xrange(s0):
        for j in xrange(s0):
            c = float(i)
            d = float(j-i)/sx
            val  = 0.0
            for n in xrange(sx):
                val += sub[int(round(c)),x[n]]
                c += d
            res[i,j] = val
            
            
def minimumLineInArray(arr, relative=False):
    '''
    find closest minimum position next to middle line
    relative: return position relative to middle line
    '''
    s0,s1 = arr.shape[:2]

    x = np.linspace(0,s1-1,min(100,s1), dtype=int)
    res = np.empty((s0,s0),dtype=float)

    _lineSumXY(x,res, arr)

    i,j = np.unravel_index(np.argmin(res), res.shape)

    if not relative:
        return i,j

    hs = s0/2
    dy0 = 2*(hs-i) +2#...no idea why +2
    dy1 = 2*(hs-j) +2
    return dy0,dy1



if __name__ == '__main__':
    import pylab as plt
    
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
    
    #plot
    plt.figure('given array')
    plt.imshow(arr)
    
    plt.figure('found line')
    plt.imshow(arr)
    plt.plot(xfit,yfit,  linewidth=10,linestyle='--')
    
    plt.show()
