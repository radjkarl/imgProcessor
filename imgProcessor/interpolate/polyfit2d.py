#code origin: http://stackoverflow.com/a/7997925
#grid method added + debug
import itertools
import numpy as np



def polyfit2d(x, y, z, order=3):
    '''
    fit unstructured data 
    '''
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(list(range(order+1)), list(range(order+1)))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m = np.linalg.lstsq(G, z)[0]
    return m


def polyval2d(x, y, m, dtype=None):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(list(range(order+1)), list(range(order+1)))
    z = np.zeros_like(x, dtype=dtype)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z


def polyfit2dGrid(arr, mask=None, order=3, replace_all=False,
                  copy=True, outgrid=None):
    '''
    replace all masked values with polynomial fitted ones
    '''
    s0,s1 = arr.shape
    if mask is None:
        if outgrid is None:
            y,x = np.mgrid[:float(s0),:float(s1)]
            p = polyfit2d(x.flatten(),y.flatten(),arr.flatten(),order)
            return polyval2d(x,y, p, dtype=arr.dtype)
        mask = np.zeros_like(arr, dtype=bool)
    elif mask.sum() == 0 and outgrid is None:
        return arr
    valid = ~mask
    y,x = np.where(valid)
    z = arr[valid]
    p = polyfit2d(x,y,z,order)
    if outgrid is not None:
        yy,xx = outgrid
    else:
        if replace_all:
            yy,xx = np.mgrid[:float(s0),:float(s1)]
        else:
            yy,xx = np.where(mask)
    new = polyval2d(xx,yy, p, dtype=arr.dtype)
    
    if outgrid is not None or replace_all:
        return new
    
    if copy:
        arr = arr.copy()
    arr[mask] = new
    return arr
 
    

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    shape=100,150
    arr = np.fromfunction(lambda x,y: (2*x-50)**2+1.9*(y-50)**2, shape)
    arr/=arr.max()
    mask = arr > 0.2

    arrin = arr.copy()
    arrin[mask] = np.nan

    arrout = polyfit2dGrid(arrin.copy(), mask, order=2)

    if 'no_window' not in sys.argv:
        plt.figure('original - structured data')
        plt.imshow(arr)
        plt.colorbar()
    
        plt.figure('input')
        plt.imshow(arrin)
        plt.colorbar()
    
        plt.figure('fit')
        plt.imshow(arrout)
        plt.colorbar()
    
        plt.show()
    
    
    # Generate Data...
    numdata = 100
    x = np.random.random(numdata)
    y = np.random.random(numdata)
    z = x**2 + y**2 + 3*x**3 + y + np.random.random(numdata)

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    if 'no_window' not in sys.argv:
        plt.figure('unstructured data')
    
        plt.imshow(zz, extent=(x.min(), y.max(), x.max(), y.min()))
        plt.scatter(x, y, c=z)
        plt.show()