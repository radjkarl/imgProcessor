import numpy as np
from numba import jit
from scipy.ndimage.filters import maximum_filter
from skimage.morphology import skeletonize
from skimage.morphology import remove_small_objects


NEIGHBOURS = np.array([[1,0],[0,1],[-1,0],[0,-1], #direct
                       [1,1],[-1,1],[-1,-1],[1,-1]#diagonals
                       ],    dtype=np.int8)


def polylinesFromBinImage(img, minimum_cluster_size=6, 
                         remove_small_obj_size=3,
                         reconnect_size=3,
                         max_n_contours=None, max_len_contour=None,
                         copy=True):
    '''
    return a list of arrays of un-branching contours
    
    img -> (boolean) array 
    
    optional:
    ---------
    minimum_cluster_size -> minimum number of pixels connected together to build a contour

    ##search_kernel_size -> TODO
    ##min_search_kernel_moment -> TODO
    
    numeric:
    -------------
    max_n_contours -> maximum number of possible contours in img
    max_len_contour -> maximum contour length
    
    '''
    assert minimum_cluster_size > 1
    assert reconnect_size%2,'ksize needs to be odd'

    #assert search_kernel_size == 0 or search_kernel_size > 2 and search_kernel_size%2, 'kernel size needs to be odd'
    #assume array size parameters, is not given:
    if max_n_contours is None:
        max_n_contours = max(img.shape)
    if max_len_contour is None:
        max_len_contour = sum(img.shape[:2])
    #array containing coord. of all contours:
    contours = np.zeros(shape=(max_n_contours,max_len_contour,2), 
                        dtype=np.uint16)# if not search_kernel_size else np.float32)


    if img.dtype != np.bool:
        img = img.astype(bool)
    elif copy:
        img = img.copy()

    if remove_small_obj_size:
        remove_small_objects(img, remove_small_obj_size, 
                         connectivity=2, in_place=True)
    if reconnect_size:
        #remove gaps
        maximum_filter(img,reconnect_size,output=img)
        #reduce contour width to 1
        img = skeletonize(img)

    n_contours = _populateContoursArray(img, contours, minimum_cluster_size)
    contours = contours[:n_contours]
    
    l = []
    for c in contours:
        ind = np.zeros(shape=len(c), dtype=bool)
        _getValidInd(c, ind)
        #remove all empty spaces:
        l.append(c[ind])
    return l


@jit(nopython=True)
def _getValidInd(c, ind):
    #only return indices of points that change the contours behaviour
    #all successive points will be removed
    gx = c.shape[0]

    px = c[1,0]
    py = c[1,1]

    dx = px-c[0,0]
    dy = py-c[0,1]
    #first point:
    ind[0]=True
    #for all following points:
    for j in xrange(2,gx-1):
        if c[j,0] == 0 and c[j,1] == 0:
            #last point
            ind[j-1]=True
            break
        npx = c[j,0]
        npy = c[j,1]
        ndx = npx-px
        ndy = npy-py
        if not (ndx == dx and ndy == dy):
            #valid new point
            ind[j-1]=True

            dx = ndx
            dy = ndy
        px = npx
        py = npy


@jit(nopython=True)
def _populateContoursArray(img, contours, minimum_cluster_size):
    # fill [contours]
    # a contour is only build by connected clusters
    gx = img.shape[0]
    gy = img.shape[1]
    n = 0
    pos = 0
    c0 = contours.shape[0]
    c1 = contours.shape[1]

    for i in xrange(gx):
        for j in xrange(gy):
            
            #found initial point:
            if img[i,j]:
                i_init = i
                j_init = j
                found = True
                img[i,j] = 0
                contours[n, pos,0] = j
                contours[n, pos,1] = i
                pos += 1
                
                #goto both directions
                for _ in xrange(2):
                    i = i_init
                    j = j_init
                    
                    while found:
                        #try to find all neighbouring points     
                        found = False

                        for neigh in xrange(8):
                            ii,jj = NEIGHBOURS[neigh]
                            i2 = i+ii
                            j2 = j+jj 
                            #out of boundary:
                            if i2 < 0 or j2 < 0 or i2 >= gx or j2 >= gy:
                                continue
                             
                            if img[i2,j2]:
                                #found new neighbour:
                                found = True
                                i = i2
                                j = j2
                                img[i,j]=0
                                break

                        if found:
                            #save x,y position of current contour point:
                            contours[n, pos,0] = j
                            contours[n, pos,1] = i
                            pos += 1
                            if pos == c1:
                                #contour too long
                                break

                    #contour too small:
                    if pos < minimum_cluster_size:
                        n-=1
                    pos = 0
                    n += 1
                    #contours array is filled - i have to stop here:
                    if n == c0:
                        return n-1
    return n



if __name__ == '__main__':
    import sys
    import pylab as plt
    import cv2
    s0,s1 = 100,100
    arr = np.zeros((s0,s1))
    #draw shape:
    arr[10:70,20]=1
    arr[30,2:99]=1
    #ellipse
    cv2.ellipse(arr, (20,20), (30,30), 0, 10, 200, 1)
    #remove few points from ellipse:
    arr[30:50,30:50] *= np.random.rand(20,20)>0.2
    #add noise
    arr += np.random.rand(s0,s1)>0.95
    contours = polylinesFromBinImage(arr)

    if 'no_window' not in sys.argv:
        plt.figure(0)
        plt.imshow(arr, interpolation='none')
    
        plt.figure(1)
        for n, c in enumerate(contours):
            if len(c)>1:
                x = c[:,0]
                y = c[:,1]
                plt.plot(x,y, linewidth=3)
                plt.text(x[-1], y[-1], str(n+1))
        plt.imshow(arr, interpolation='none')
        plt.set_cmap('gray')
        plt.show()    
    
