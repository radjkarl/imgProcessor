import numpy as np

class LinearInterpolateImageStack(object):
    
    def __init__(self, imgs, z=None, dynamic=False):
        '''
        dynamic -> whether imgs and/or z are callable
                   and data is freshly fetched every time this class is called
        '''
        if z is None and not dynamic:
            self.z = np.arange(len(imgs))
        else:
            self.z = z
        self.imgs = imgs
        
        if dynamic:
            self._call = self._call_dynamic
        else:
            self._call = self._call_static


    def __call__(self, val):
        return self._call(val)

            
    def _call_static(self, val):
        return self._interpolate(val, self.z, self.imgs)
        
    
    def _call_dynamic(self, val):
        imgs = self.imgs()
        if self.z is None:
            z = np.arange(len(imgs))
        else:
            z = self.z()
        return self._interpolate(val, z, imgs)
       
            
    @staticmethod
    def _interpolate(val, z, imgs):
        if val<=z[0]:
            return imgs[0]
        if val>=z[-1]:
            return imgs[-1]    
        ind = np.argmax(z>val)
        z0,z1 = z[ind-1:ind+1]
        dz = z1-z0
        f = (val-z0)/dz
        return (1-f)*imgs[ind-1] + f*imgs[ind]
        


if __name__ == '__main__':
    import sys
    import pylab as plt

    r2,r3 = 10,15
    c = 50
    n = 10
    z_values = np.linspace(0.1,1,n)

    imgs = np.fromfunction(lambda z,x,y: np.sin(z*x-50)
                                        +z*np.cos(y-c), (n,r2,r3))
    
    I = LinearInterpolateImageStack(imgs, z_values)

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

    