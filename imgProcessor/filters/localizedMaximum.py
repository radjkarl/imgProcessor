import numpy as np
from numba import jit


def localizedMaximum(img, thresh=0, min_increase=0, max_length=0, dtype=bool):
    '''
    Returns the local maximum of a given 2d array
    
    
    thresh -> if given, ignore all values below that value
    
    max_length -> limit length between value has to vary  > min_increase
    
    >>> a = np.array([[0,1,2,3,2,1,0], \
                      [0,1,2,2,3,1,0], \
                      [0,1,1,2,2,3,0], \
                      [0,1,1,2,1,1,0],  \
                      [0,0,0,1,1,0,0]])
    
    >>> print localizedMaximum(a, dtype=int)
    [[0 1 1 1 0 1 0]
     [0 0 0 0 1 0 0]
     [0 0 0 1 0 1 0]
     [0 0 1 1 0 1 0]
     [0 0 0 1 0 0 0]]
    '''
    #because numba cannot create arrays:
    out = np.zeros(shape=img.shape, dtype=dtype)
    #first iterate all rows:
    _calc(img, out, thresh, min_increase, max_length)
    #that all columns:
    _calc(img.T, out.T, thresh, min_increase, max_length)
    return out


@jit(nopython=True) 
def _calc(img, out, thresh, min_increase, max_length):
    g0 = img.shape[0]
    g1 = img.shape[1]

    found_peak = False
    lmx = thresh #value of localized maximum
    pmx = 0 #position of loc. mx

    for i in xrange(g0):
        for j in xrange(g1):
            px = img[i,j]
            if not found_peak and px > thresh:
                if not max_length:
                    found_peak = True
                else:#go back and check increase
                    for jj in xrange(j-1,max(-1,j-1-max_length),-1):
                        if px - img[i,jj] > min_increase:
                            found_peak = True
                            break
            if found_peak:
                #move local maximum:
                if px > lmx:
                    lmx  = px 
                    pmx = j

                elif lmx - px >min_increase:
                    if not max_length or j - pmx <= max_length:
                        #at peaks end:
                        out[i,pmx] = 1 
                    #reset:
                    found_peak = False
                    lmx = thresh
        lmx = thresh
                    


if __name__ == '__main__':
#     import doctest
#     doctest.testmod()
    import sys
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    from imgProcessor.imgIO import imread
    from scipy.ndimage.filters import maximum_filter

    p = PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence')


    img = imread(p.join('EL_cell_cracked.png'), 'gray')
    
    bn = maximum_filter(#<-- make lines bold
            localizedMaximum(-img, thresh=30, min_increase=10, max_length=10)
            ,3)

    if 'no_window' not in sys.argv:
        plt.figure('image')
        plt.imshow(img)
        plt.colorbar()
        plt.figure('binarized')
        plt.imshow(bn)        
        plt.show()