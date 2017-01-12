from __future__ import division

import numpy as np
import cv2
from scipy.optimize import brent
from imgProcessor.interpolate.polyfit2d import polyfit2dGrid
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.filters.edges import roberts
from imgProcessor.imgSignal import signalMinimum
from skimage.filters.thresholding import threshold_otsu


def _fit(offs, y0,y1): 
    return np.abs(_shift(offs, y0,y1)-y1).mean()
    
def _shift(offs, y0,y1):
    x = np.arange(len(y0))
    return np.interp(x+offs,x,y0)


def subPixelAlignment(img, ref, grid=None, nCells=5,
                       smoothOrder=None, maxDev=5, concentrateNNeighbours=2, maps=None):
    '''
    align images showing almost the same content 
    with only a small positional error
    (few pixels) to top of each other.
    Remaining spatial deviation < 1 px
    
    smoothOrder [1....5] order of polynomial fit to smooth offsets
    maxDev (int) --> maximum pixel deviation
           higher values will be excluded
    '''
    if maps is None:
        maps = subPixelAlignmentMaps(img, ref, grid, nCells,
                       smoothOrder, maxDev)
    mapX,mapY = maps
    
#     import pylab as plt
#     plt.imshow(cv2.remap(img, mapY, mapX, interpolation=cv2.INTER_LANCZOS4,
#                   borderValue=0, borderMode=cv2.BORDER_REFLECT)-ref)
#     plt.show()
    
    return cv2.remap(img, mapY, mapX, interpolation=cv2.INTER_LANCZOS4,
                  borderValue=0, borderMode=cv2.BORDER_REFLECT)



def subPixelAlignmentMaps(img, ref, grid=None, nCells=5,
                       smoothOrder=None, maxDev=None,concentrateNNeighbours=2):
#     if binarize:
#         
#         img = (img > threshold_otsu(img)).astype(np.float32)
#         ref = (ref > threshold_otsu(ref)).astype(np.float32)
#         import pylab as plt
#         plt.imshow(img)
#         plt.figure(2)
#         plt.imshow(ref)
 
#         plt.show()
    s0,s1 = img.shape
    aR = s0/s1
    if grid is None:
        if aR < 1:
            grid = nCells,int(nCells/aR)  
        else:
            grid = int(nCells*aR), nCells   
    if maxDev is None :
        #set maxDev to a 15th of cellSize:
        maxDev = np.mean([s0/grid[0], s1/grid[1]]) / 15

#     img = img / gaussian_filter(img,11)
#     ref = ref / gaussian_filter(ref,11)


#     img = roberts(img)
#     ref = roberts(ref)

    #average subsections through rescaling:
    i0a = cv2.resize(img,(grid[1], s0), interpolation=cv2.INTER_AREA).T
    i0b = cv2.resize(img,(s1, grid[0]), interpolation=cv2.INTER_AREA)

    i1a = cv2.resize(ref,(grid[1], s0), interpolation=cv2.INTER_AREA).T
    i1b = cv2.resize(ref,(s1, grid[0]), interpolation=cv2.INTER_AREA)
    
    
    
    
    
#     i0a = cv2.Scharr(i0a, cv2.CV_32F, 0, 1)
    
#     i0a = np.gradient(i0a, axis=1)
#     i1a = np.gradient(i1a, axis=1)
#     i0b = np.gradient(i0b, axis=0)
#     i1b = np.gradient(i1b, axis=0)    
    
#     import pylab as plt
# #     plt.plot(i0a.T)
#     plt.plot(np.gradient(i0a, axis=1).T)
#     plt.show()
    
    offset = np.full(shape=(grid[0], grid[1], 2),
                     fill_value=np.nan, dtype=np.float32)
    
    #pixel indices:
    xr = np.linspace(0,s1,grid[1]+1, dtype=int)
    yr = np.linspace(0,s0,grid[0]+1, dtype=int)

    #vertical shift
    for i in range(grid[1]):
        ri0a = i0a[i]
        ri1a = i1a[i]
        for j in range(grid[0]):
            
            k = min(grid[0]-j-1,min(j,concentrateNNeighbours))
#             if k ==0:#!= concentrateNNeighbours:
#                 continue
#             print(k)

            y0 = ri0a[yr[j-k]:yr[j+1+k]]
            y1 = ri1a[yr[j-k]:yr[j+1+k]]
            

            offs = brent (lambda offs: _fit(offs, y0, y1))
            offset[j,i,0] = offs
        
#             import pylab as plt
#             plt.plot(_shift(offs, y0, y1))
#             plt.plot(y1)
#             plt.show()

    #horizontal shift
    for i in range(grid[0]):
        ri0b = i0b[i]
        ri1b = i1b[i]
        for j in range(grid[1]):

            k = min(grid[1]-j-1,min(j,concentrateNNeighbours))
#             if k ==0:#!= concentrateNNeighbours:
#                 continue

            y0 = ri0b[xr[j-k]:xr[j+1+k]]
            y1 = ri1b[xr[j-k]:xr[j+1+k]]

            offs = brent (lambda offs: _fit(offs, y0, y1))
            offset[i,j,1] = offs


#             plt.plot(y0)
#             plt.plot(y1)
#             x = np.arange(len(y0))
# # 
#             y3 = np.interp(x+offs,x,y0)
#              
#             plt.plot(y3, 'o-')
#             plt.show()


    #exclude erroneous results:
    


    #smooth OPTIONAL
    if smoothOrder != 0:
        if smoothOrder is None:
            smoothOrder = 2
        so = smoothOrder
        for _ in range(so):
            try: 
                offset[...,1] = polyfit2dGrid(
                            offset[...,1], 
                            mask=np.logical_or(np.isnan(offset[...,1]),
                                np.abs(offset[...,1])>maxDev),
                            order=so, 
                            replace_all=True)
                break
            except: so-=1
            
        so = smoothOrder
        for _ in range(so+1):
            try: 
                offset[...,0] = polyfit2dGrid(
                            offset[...,0], 
                            mask=np.logical_or(np.isnan(offset[...,0]),
                                               np.abs(offset[...,0])>maxDev),
                            order=so, 
                            replace_all=True)
                
                break
            
            except: 
                print(11)
                so-=1


#     import pylab as plt
#     plt.imshow(offset[...,1], clim=(-10,10), interpolation='none')
#     plt.colorbar()
#     plt.figure(2)
#     plt.imshow(offset[...,0], interpolation='none')
#     plt.colorbar()
#     plt.show()

#     import pylab as plt
#     plt.imshow(img-ref)
#     plt.colorbar()
#     plt.figure(2)
#     plt.imshow(offset[...,0])
#     plt.colorbar()
#     plt.show()



#     offset[...,0] = median_filter(offset[...,0],3)
#     offset[...,1] = median_filter(offset[...,1],3)


    import pylab as plt
    plt.imshow(offset[...,0], interpolation='none', clim=(-5,5))
    plt.colorbar()
    plt.figure(2)
    plt.imshow(offset[...,1], interpolation='none', clim=(-5,5))
    plt.colorbar()
    plt.show()

    #resize to image size:
    mapX = cv2.resize(offset[...,0],(s1, s0), 
                      interpolation=cv2.INTER_LANCZOS4)
    mapY = cv2.resize(offset[...,1],(s1, s0), 
                      interpolation=cv2.INTER_LANCZOS4)
#     mapX[:]=1
    #pixel indices:
    oX,oY = np.mgrid[:s0,:s1]

#     plt.imshow(mapX, interpolation='none')
#     plt.colorbar()
#     plt.show()

    mapX+=oX
    mapY+=oY
    return mapX,mapY
            
if __name__ == '__main__':
    from fancytools.os.PathStr import PathStr
    from imgProcessor.imgIO import imread
    import imgProcessor
    import pylab as plt   
    import sys

    p = PathStr(imgProcessor.__file__).dirname().join(
                'media','electroluminescence')
    i0 = imread(p.join('EL_mod_pos1.jpg'))
    i1 = imread(p.join('EL_mod_pos2.jpg'))
    
    i0fit = subPixelAlignment(i0, i1)

    if 'no_window' not in sys.argv:

        f, ax = plt.subplots(2,4)
        
        ax[0,0].set_title('EL image 0')
        ax[0,0].imshow(i0, interpolation='none')  
        ax[0,1].set_title('EL image 1')
        ax[0,1].imshow(i1, interpolation='none')  
    
        clim = -50,50
        d0 = i0.astype(float)-i1
        ax[0,2].set_title('image difference')
        ax[0,2].imshow(d0, interpolation='none', clim=clim)  
        ax[0,3].set_title('detail')
        ax[0,3].imshow(d0[:380,2900:], interpolation='none', clim=clim)  
        
        ax[1,0].set_title('EL image 0 corrected')
        ax[1,0].imshow(i0fit, interpolation='none')  
        f.delaxes(ax[1,1])
        d1 = i0fit.astype(float)-i1
        ax[1,2].imshow(d1, interpolation='none', clim=clim)  
        ax[1,3].imshow(d1[:380,2900:], interpolation='none', clim=clim)  
     
        plt.show()