from __future__ import division

import numpy as np
from scipy.interpolate.interpolate import RegularGridInterpolator

#TODO: this method is very slow ... speed up
#TODO: implement cubic interpolation as well
class InterpolateImageStack(object):
    def __init__(self, imgs, z_values=None,
                       method='linear', bounds_error=True, 
                       fill_value=np.nan):
        '''
        imgs -> image stack (only grayscale)
        
        methods --> 'linear',  'cubic'
        
        optional:
        z_values -> depth values (e.g. exposure times)
                     if not given assumed to be 
                     linear and (0...len(imgs))
        '''
        n = len(imgs)
        self.s = s0,s1 = imgs[0].shape
        y,x = np.arange(0,s0),np.arange(0,s1)
        
        if z_values is None:
            z_values = np.arange(0,n)
#         if method == 'linear':
#             self.interpolator = LinearInterpolateImageStack(z_values, imgs)
#             InterpolateImageStack.__call__ = self.interpolator.__call__
#         else:
        self.interpolator = RegularGridInterpolator((z_values, y, x), imgs, 
                                method=method, 
                                bounds_error=bounds_error, 
                                fill_value=fill_value)
        Y,X = np.mgrid[:s0,:s1]
        self.pts = np.empty(shape=(s0*s1,3))
        self.pts[:,1] = Y.flatten()
        self.pts[:,2] = X.flatten()
    
    
    def __call__(self, z):
        self.pts[:,0]  = z
        return self.interpolator(self.pts).reshape(self.s)


# TODO: more advanced class with same functionality
# also in 'LinearInterpolateImageStack
class LinearInterpolateImageStack(object):
    
    def __init__(self, imgs, z=None):
        if z is None:
            self.z = np.arange(len(imgs))
        else:
            self.z = np.asfarray(z)
        self.imgs = np.asfarray(imgs)
        
    def __call__(self, val):
        ind = np.argmax(self.z>val)
        if ind == 0:
            return self.imgs[0]
        elif ind == -1:
            return self.imgs[-1]
        z0,z1 = self.z[ind-1:ind+1]
        dz = z1-z0
        f = (val-z0)/dz
        return f*self.imgs[ind-1] + (1-f)*self.imgs[ind]
        


if __name__ == '__main__':
    import sys
    import pylab as plt
    from time import time

    res2,res3 = 500,700
    c = 500
    n = 10
    f=100
    z_values = np.linspace(0,1,n)

    imgs = np.fromfunction(lambda z,x,y: np.sin(z*x/f-50)
                                        +z*np.cos(y/f-c), (n,res2,res3))
    t0 = time()
    I = InterpolateImageStack(imgs, z_values)

    zz = np.sort(np.random.rand(4))
    #interpolated images:
    ii = [I(z) for z in zz]
    print(time()-t0)
    
    if 'no_window' not in sys.argv:
        f,ax = plt.subplots(2,2)
        f.suptitle('Interpolated images at random positions in between 0-1')
        ax[0,0].set_title(zz[0])
        ax[0,0].imshow(ii[0], interpolation='none')
        
        ax[0,1].set_title(zz[1])
        ax[0,1].imshow(ii[1], interpolation='none')
        
        ax[1,0].set_title(zz[2])
        ax[1,0].imshow(ii[2], interpolation='none')
        
        ax[1,1].set_title(zz[3])
        ax[1,1].imshow(ii[3], interpolation='none')
        
        plt.show()    

    