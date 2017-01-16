from __future__ import division

import cv2
import numpy as np
import warnings

from scipy.optimize import brent
# from scipy.ndimage.filters import gaussian_filter

from imgProcessor.filters.fastFilter import fastFilter
from imgProcessor.interpolate.polyfit2d import polyfit2dGrid
# from imgProcessor.interpolate.interpolate2dStructuredFastIDW import interpolate2dStructuredFastIDW



def _fit(offs, y0,y1): 
    #return absolute average deviation
    dy = np.abs(_shift(offs, y0,y1)-y1)
    return np.nanmean(dy)


def _shift(offs, y0,y1):
    #shift y0 in x direction
    x = np.arange(len(y0))
    return np.interp(x+offs,x,y0)



def iterativeSubPixelAligmentMaps(img, ref, niter=10,nitermin=1,
                                  maxLastDev=0.1,maxDev=30,
                                  #borderValue=0, borderMode=cv2.BORDER_CONSTANT,
                                  borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan,
                                  **kwargs):
    '''
    like subPixelAlignment
    but repeating process till dev<maxDev
    '''
    res = None
    img2 = img
    n=1
    lastoff = 100
    
    while True:
        offset = _findOffset(img2, ref, maxDev=maxDev,**kwargs)
        
        newoff = np.abs(offset).mean()
        if newoff>lastoff:
            break
        lastoff = newoff

        if n>nitermin and np.abs(offset).mean()<maxLastDev or n>niter:
            break
        n+=1
     
        if res is None:
            #first time:
            res = offset  
        else:
            #all following times:
            res += offset
  
        mapX, mapY = _mkMap(img, res)
        
        img2 =  cv2.remap(img, mapY, mapX, 
                          interpolation=cv2.INTER_LANCZOS4,#cv2.INTER_NEAREST
                          borderMode=borderMode,borderValue=borderValue)
       
    
    return img2, (mapX, mapY), res    


def subPixelAlignment(img, ref, grid=None, nCells=5,
                       smoothOrder=5, maxDev=5, 
                       concentrateNNeighbours=2, maps=None):
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

    return cv2.remap(img, mapY, mapX, interpolation=cv2.INTER_LANCZOS4,
                  borderValue=0, borderMode=cv2.BORDER_REFLECT)


def _findOffset(img, ref, grid=None, nCells=5,
                       smoothOrder=5, maxDev=None,
                       maxGrad=None,
                       concentrateNNeighbours=2):
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
    if maxGrad is None :
        maxGrad = (2*maxDev)/max(grid)

    #average subsections through rescaling:
    i0a = np.asfarray(cv2.resize(img,(grid[1], s0), interpolation=cv2.INTER_AREA).T)
    i0b = np.asfarray(cv2.resize(img,(s1, grid[0]), interpolation=cv2.INTER_AREA))

    i1a = np.asfarray(
            cv2.resize(ref,(grid[1], s0), interpolation=cv2.INTER_AREA).T )
    i1b = np.asfarray(
            cv2.resize(ref,(s1, grid[0]), interpolation=cv2.INTER_AREA) )
  
    offset = np.full(shape=(grid[0], grid[1], 2),
                     fill_value=np.nan, dtype=np.float32)
    
    #pixel indices:
    xr = np.linspace(0,s1,grid[1]+1, dtype=int)
    yr = np.linspace(0,s0,grid[0]+1, dtype=int)

    #vertical shift
    for i in range(grid[1]):
        ri0a = i0a[i]
        ri1a = i1a[i]
#         print(ri0a)
        for j in range(grid[0]):
            
            k = min(grid[0]-j-1,min(j,concentrateNNeighbours))

            y0 = ri0a[yr[j-k]:yr[j+1+k]]
            y1 = ri1a[yr[j-k]:yr[j+1+k]]

            if np.isfinite(y0).any() and np.isfinite(y1).any():

                #scale:
                y0 = y0-np.nanmin(y0)
                y1 = y1-np.nanmin(y1)    
                y0 = y0/np.nanmean(y0)
                y1 = y1/np.nanmean(y1)

                offs = brent (lambda offs: _fit(offs, y0, y1))
                offset[j,i,0] = offs

    #horizontal shift
    for i in range(grid[0]):
        ri0b = i0b[i]
        ri1b = i1b[i]
        for j in range(grid[1]):

            k = min(grid[1]-j-1,min(j,concentrateNNeighbours))

            y0 = ri0b[xr[j-k]:xr[j+1+k]]
            y1 = ri1b[xr[j-k]:xr[j+1+k]]

            if np.isfinite(y0).any() and np.isfinite(y1).any():
                #scale:
                y0 = y0-np.nanmin(y0)
                y1 = y1-np.nanmin(y1)    
                y0 = y0/np.nanmean(y0)
                y1 = y1/np.nanmean(y1)
                with warnings.catch_warnings():
                    offs = brent (lambda offs: _fit(offs, y0, y1))
                offset[i,j,1] = offs

    o0,o1 = offset[...,0].copy(), offset[...,1].copy()

    def getMask(oo):
        with np.errstate(invalid='ignore'):
            oo[np.abs(oo)>maxDev] = np.nan
            ff = fastFilter(oo.copy(), ksize=3, fn='nanmedian')
        return np.logical_or(np.isnan(oo), 
                             #spatial change too high:
                             np.logical_and(ff>1, np.abs((oo-ff)/ff)>1) )

    #smooth OPTIONAL
    if smoothOrder != 0:
        so = smoothOrder
        #if smooth order is too high to fit: reduce sequentially:
        for _ in range(so):
            try: 
                o1 = polyfit2dGrid(
                            o1, 
                            mask=getMask(o1),
                            order=so, 
                            replace_all=True)
                break
            except: so-=1
             
        so = smoothOrder
        for _ in range(so+1):
            try: 
                o0 = polyfit2dGrid(
                            o0, 
                            mask=getMask(o0),
                            order=so, 
                            replace_all=False)
                 
                break
            except: 
                so-=1

#ALTERNATIVE:
#     interpolate2dStructuredFastIDW(o0,getMask(o0))
#     interpolate2dStructuredFastIDW(o1,getMask(o1))  

#     o0 = gaussian_filter(median_filter(o0, 3),0.7)
#     o1 = gaussian_filter(median_filter(o1, 3),0.7)

    offset[...,0]=o0
    offset[...,1]=o1
    
    return offset 
    

def subPixelAlignmentMaps(img, *args, **kwargs):
    offset = _findOffset(img, *args, **kwargs)
    return _mkMap(img, offset)


def _mkMap(img, offset):
    s0,s1 = img.shape

    #resize to image size:
    mapX = cv2.resize(offset[...,0],(s1, s0), 
                      #gives better results with LINEAR than with LANCZOS
                      interpolation=cv2.INTER_LINEAR)#LANCZOS4)
    mapY = cv2.resize(offset[...,1],(s1, s0), 
                      interpolation=cv2.INTER_LINEAR)
    #pixel indices:
    oX,oY = np.mgrid[:s0,:s1]

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