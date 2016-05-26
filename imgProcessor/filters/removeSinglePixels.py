from numba import jit


@jit(nopython=True)
def removeSinglePixels(img):
    '''
    img - boolean array
    remove all pixels that have no neighbour
    '''

    gx = img.shape[0]
    gy = img.shape[1]

    for i in xrange(gx):
        for j in xrange(gy):
            
            if img[i,j]:
                
                found_neighbour = False
                for ii in xrange(max(0,i-1),min(gx,i+2)):
                    for jj in xrange(max(0,j-1),min(gy,j+2)):
 
                        if ii == i and jj == j:
                            continue
                        
                        if img[ii,jj]:
                            found_neighbour = True
                            break
                    if found_neighbour:
                        break

                if not found_neighbour:
                    img[i,j] = 0



if __name__ == '__main__':
    import sys
    import numpy as np
    import pylab as plt
    
    arr = np.random.rand(100,100)>0.9
    
    arr2 = arr.copy()
    removeSinglePixels(arr2)

    if 'no_window' not in sys.argv:
        plt.figure(1)
        plt.imshow(arr, interpolation='none')
    
        plt.figure(2)
        plt.imshow(arr2, interpolation='none')
        
        plt.show()