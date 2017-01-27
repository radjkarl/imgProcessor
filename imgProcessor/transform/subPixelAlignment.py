from __future__ import division

import cv2
import numpy as np
import warnings

from scipy.optimize import brent
# from scipy.ndimage.filters import gaussian_filter

from imgProcessor.filters.fastFilter import fastFilter
from imgProcessor.interpolate.polyfit2d import polyfit2dGrid
from scipy.ndimage.filters import minimum_filter, gaussian_filter, median_filter
from imgProcessor.interpolate.interpolate2dStructuredFastIDW import interpolate2dStructuredFastIDW
from skimage.filters.edges import roberts
# from imgProcessor.interpolate.interpolate2dStructuredFastIDW import interpolate2dStructuredFastIDW


#traceback #RuntimeWarning: Mean of empty slice#
# import warnings
# warnings.simplefilter("error")



def _fit(offs, x, y0,y1): 
    #return absolute average deviation
    dy = np.abs(_shift(offs, x, y0)-y1)
    return dy.mean()
#     if 
#     return np.nanmean(dy)#dy.mean()#'
    try:
        return np.nanmean(dy)
    except RuntimeWarning:
        return np.NaN 

def _shift(offs, x, y0):
    #shift y0 in x direction
#     x = np.arange(len(y0))
#     ind = ~np.logical_or(np.isnan(y1),np.isnan(y1))
#     x = x[ind]
#     y0 = y0[ind]
#     y1 = y1[ind]
    return np.interp(x+offs,x,y0, left=0, right=0)


# def addBorder(img, size=50):
#     s0,s1 = img.shape
#     img2 = np.zeros(shape=(s0+size*2,s1+size*2), dtype=img.dtype )
#     img2[size:-size, size:-size]=img
#     return img2
# 
# def delBorder(img, size=50):
#     return img[size:-size, size:-size]



def iterativeSubPixelAligmentMaps(img, ref, niter=10,nitermin=3,
                                  maxLastDev=0.1,maxDev=30,
                                  #borderValue=0, borderMode=cv2.BORDER_CONSTANT,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=np.nan,
                                  **kwargs):
    '''
    like subPixelAlignment
    but repeating process till dev<maxDev
    '''
    
#     img = addBorder(img)
#     ref = addBorder(ref)



#     img = img/img.max()
#     ref = ref/ref.max()
#     print(img.min(), img.max())
#     img = roberts(img)
#     ref = roberts(ref)
    
    res = None
    img2 = img
    n=1
    lastoff = 100
#     smoothOrder = 0
#     if 'smoothOrder' in kwargs:
#         smoothOrder = kwargs.pop('smoothOrder')



    
    while True:
        offset = _findOffset(img2, ref, maxDev=maxDev,**kwargs)
        
#         import pylab as plt
#         plt.figure(1)
#         plt.imshow(offset[...,0], interpolation='none')
#         plt.colorbar()
#         plt.figure(2)
#         plt.imshow(offset[...,1], interpolation='none')
#         plt.colorbar()
#  
#         plt.show()
        newoff = np.abs(offset).mean()
        print(newoff)

        if n>nitermin and (newoff>lastoff or newoff<maxLastDev):
            break
        lastoff = newoff


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
        if n>niter:
            break
#         import pylab as plt  
#         plt.figure(n)
#         o = offset.copy()
#           
#         plt.imshow(img2, interpolation='none')#img2-ref)#(o[...,0]**2+o[...,1]**2)**0.5)#img2-ref)
#         plt.colorbar()
#     plt.show()
    
    return img2, (mapX, mapY), res    


def subPixelAlignment(img, ref, maps=None, **kwargs):
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
        maps = subPixelAlignmentMaps(img, ref,**kwargs)
    mapX,mapY = maps

    return cv2.remap(img, mapY, mapX, interpolation=cv2.INTER_LANCZOS4,
                  borderValue=0, borderMode=cv2.BORDER_REFLECT)


def _findOffset(img, ref, grid=None, nCells=5,
                       method='fit',#'smooth'
                       #smoothOrder=5, 
                       maxDev=None,
                       maxGrad=None,
                       concentrateNNeighbours=2,
                       #ignoreFirstNCells=0
                       ):
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



    def _stuff(y0,y1):
        ind = np.logical_and(np.isfinite(y0),np.isfinite(y1))
        if ind.any():
            
            x = np.arange(len(y0))
            x = x[ind]
            y0 = y0[ind]
            y1 = y1[ind]
#             print(y0.min(), y0.mean())
            y0-=y0.min()
            y0/=y0.max()
            y1-=y1.min()
            y1/=y1.max()        
        
        #if  np.isfinite(y0).any() and np.isfinite(y1).any():
            #print(111)
            #scale:
#             y0 = y0-np.nanmin(y0)
#             y1 = y1-np.nanmin(y1)    
#             y0 = y0/np.nanmean(y0)
#             y1 = y1/np.nanmean(y1)
            #with warnings.catch_warnings():
            
            out = brent (lambda offs: _fit(offs, x, y0, y1),
                         brack=(-maxDev,maxDev))
            if abs(out)>maxDev:
                return np.nan
#             print(out)
#             if -2.61>out>-4:
#                 import pylab as plt
# #                 print(x)
# #                 print(y0)
# #                 print(y1)
#                 plt.plot(x,y0)
#                 plt.plot(x,y1)
#                 plt.show()
            return out
#         print(77777)
        return np.nan

    g0,g1 = grid
    #vertical shift
    for i in range(g1):
        ri0a = i0a[i]
        ri1a = i1a[i]
#         print(ri0a)
        for j in range(g0):
#             if (i<ignoreFirstNCells or j<ignoreFirstNCells 
#                 or g1-i-1<ignoreFirstNCells or g0-j-1<ignoreFirstNCells):
#                 continue
            k = min(grid[0]-j-1,min(j,concentrateNNeighbours))
#             print(j,k)
            slic = slice(yr[j-k],yr[j+1+k])
            y0 = ri0a[slic]
            y1 = ri1a[slic]
#             print(y0)
#             print(y1)
#             print(11)
#             print(111)
            offset[j,i,0] = _stuff(y0,y1)
            

                
            
#             print(222)


    #horizontal shift
    for i in range(g0):
        ri0b = i0b[i]
        ri1b = i1b[i]
        for j in range(g1):
            k = min(grid[1]-j-1,min(j,concentrateNNeighbours))
            slic = slice(xr[j-k], xr[j+1+k])
            y0 = ri0b[slic]
            y1 = ri1b[slic]


            offset[i,j,1] = _stuff(y0,y1)
#             if j==grid[1]-1 and i==6:
#                  
#                 ind = np.logical_and(np.isfinite(y0),np.isfinite(y1))
#                 x = np.arange(len(y0))
#                 x = x[ind]
#                 y0 = y0[ind]
#                 y1 = y1[ind]
#                  
#                 y0-=y0.min()
#                 y0/=y0.mean()
#                 y1-=y1.min()
#                 y1/=y1.mean()  
#                  
#                 import pylab as plt
# #                 ddd = []
# #                 for oo in np.linspace(-20,20,300):
# #                     ddd.append(np.abs(_shift(oo, x, y0)-y1).mean())
# #                 plt.plot(np.linspace(-20,20,300), ddd)
# #                 plt.show()
#                 oo = offset[i,j,1]
#                 print(oo)
#                 plt.plot(y1)
#                 plt.plot(y0)#
#                 plt.plot(_shift(oo, x, y0),'o-')
#                 plt.show()
                
                
#             if not np.isnan(y0).all() and not np.isnan(y1).all():
#                 #scale:
#                 y0 = y0-np.nanmin(y0)
#                 y1 = y1-np.nanmin(y1)    
#                 y0 = y0/np.nanmean(y0)
#                 y1 = y1/np.nanmean(y1)
#                 with warnings.catch_warnings():
#                     offs = brent (lambda offs: _fit(offs, y0, y1))
#                 offset[i,j,1] = offs

#     o0,o1 = offset[...,0].copy(), offset[...,1].copy()
#     return offset
    def getMask(oo):
        with np.errstate(invalid='ignore'):
            oo[np.abs(oo)>maxDev] = np.nan
            return np.isnan(oo)
            ff = fastFilter(oo.copy(), ksize=3, fn='nanmedian')
            return np.logical_or(np.isnan(oo), 
                             #spatial change too high:
                             np.logical_and(ff>1, np.abs((oo-ff)/ff)>1) )

    #smooth OPTIONAL
    if method=='fit':
#         maxGrad=2
        
        def _fit2(oi):
            oo = oi.copy()

#             s0 = np.sign(oo)
#             oo = s0*minimum_filter(np.abs(oo),3)
            
            #if smooth order is too high to fit: reduce sequentially:
            fit = []
            err = []
            
            for f in range(maxGrad,0,-1):
                try: 
                    oi = polyfit2dGrid(
                                oo, 
                                mask=getMask(oo),
                                order=f, 
                                replace_all=True)

                    fit.append(oi)
                    err.append(np.nanmean(np.abs(oo-oi)))
                    #break
                except: 
                    
                    fit.append(None)
                    err.append(np.nan)
            i = np.nanargmin(err)
            return fit[i]

        o1 = _fit2(offset[...,1])
        o0 = _fit2(offset[...,0])


    else:

    
#         import pylab as plt
#         plt.figure(100)
#         plt.imshow(offset[...,0], interpolation='none')
#         plt.colorbar()
#         plt.figure(200)
#         plt.imshow(offset[...,1], interpolation='none')
#         plt.colorbar()
#         plt.show()
    
        
        o0 = offset[...,0]
        o1 = offset[...,1]
    
        s0 = np.sign(o0)
        o0 = offset[...,0]
        s1 = np.sign(o1)
        
        o0 = s0*minimum_filter(np.abs(o0),3)
        o1 = s1*minimum_filter(np.abs(o1),3)
    
    
        interpolate2dStructuredFastIDW(o0,np.isnan(o0))#getMask(o0))
        interpolate2dStructuredFastIDW(o1,np.isnan(o1))#getMask(o1))  

#         o0 = median_filter(o0,3)
#         o1 = median_filter(o1,3)    
        o0 = gaussian_filter(o0,1)
        o1 = gaussian_filter(o1,1)
    offset[...,0]=o0
    offset[...,1]=o1
#     import pylab as plt
#     plt.figure(1)
#     plt.imshow(offset[...,0], interpolation='none')
#     plt.colorbar()
#     plt.figure(2)
#     plt.imshow(offset[...,1], interpolation='none')
#     plt.colorbar()
#     plt.show()


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