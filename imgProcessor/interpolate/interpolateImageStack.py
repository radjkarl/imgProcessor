import numpy as np
from scipy.interpolate.interpolate import RegularGridInterpolator



class InterpolateImageStack(object):
    def __init__(self, imgs, z_values=None,
                       method='linear', bounds_error=True, 
                       fill_value=np.nan):
        '''
        imgs -> image stack (only grayscale)
        
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
        self.interpolator = RegularGridInterpolator((z_values, y, x), imgs, 
                                    method=method, 
                                    bounds_error=bounds_error, 
                                    fill_value=fill_value)
        Y,X = np.mgrid[:s0,:s1]
        self.pts = np.empty(shape=(s0*s1,3))
        self.pts[:,1] = Y.flatten()
        self.pts[:,2] = X.flatten()
    
    
    def __call__(self, z):
        self.pts[:,0]  =z
        return self.interpolator(self.pts).reshape(self.s)



if __name__ == '__main__':
    import sys
    import pylab as plt

    r2,r3 = 10,15
    c = 50
    n = 10
    z_values = np.linspace(0.1,1,n)

    imgs = np.fromfunction(lambda z,x,y: np.sin(z*x-50)
                                        +z*np.cos(y-c), (n,r2,r3))
    
    I = InterpolateImageStack(imgs, z_values)

    zz = np.random.rand(4)
    #interpolated images:
    ii = [I(z) for z in zz]

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

    