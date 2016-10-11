import numpy as np
from numba import jit


#test different k sizes also min, ang, median
####add much more EL samples to bg comparison 
###################

def estimateBackgroundLevel(img, image_is_artefact_free=False, 
                            min_rel_size=0.05, max_abs_size=11):
    '''
    estimate background level through finding the most homogeneous area
    and take its average
    
    min_size - relative size of the examined area
    '''

    s0,s1 = img.shape[:2]
    s = min(max_abs_size, int(max(s0,s1)*min_rel_size))
    arr = np.zeros(shape=(s0-2*s, s1-2*s), dtype=img.dtype)
    
    #fill arr:
    _spatialStd(img, arr, s)
    #most homogeneous area:
    i,j = np.unravel_index(arr.argmin(), arr.shape)
    sub = img[int(i+0.5*s):int(i+s*1.5), 
              int(j+s*0.5):int(j+s*1.5)]

    return np.median(sub)


@jit(nopython=True) 
def _spatialStd(img, arr, s):
    g0 = img.shape[0]
    g1 = img.shape[1]
    hs = s/2
    #for every pixel:
    for i in xrange(s,g0-s):
        for j in xrange(s,g1-s):
            arr[i-s,j-s] = img[i-hs:i+hs, j-hs:j+hs].std()



if __name__ == '__main__':
    from scipy.ndimage.filters import gaussian_filter
    import pylab as plt
    import sys

    bg_level = 67
    obj_level = 200
    noise = 4
    i,j = 100,100
    
    arr = np.full((i,j), bg_level)

    #add object:
    arr[int(i*0.2):-int(i*0.2), int(j*0.3):int(-j*0.3)] = obj_level
    arr = gaussian_filter(arr, 7)
    
    #add noise:
    arr += (np.random.rand(i,j)*2-1)*noise

    bglevel = estimateBackgroundLevel(arr)
    
    if 'no_window' not in sys.argv:
        plt.imshow(arr)
        plt.title('bg level=%s' %bglevel)
        plt.colorbar()
        plt.show()
            
            