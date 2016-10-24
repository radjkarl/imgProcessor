import numpy as np

from imgProcessor.imgIO import imread
from imgProcessor.array.subCell2D import subCell2DFnArray, subCell2DCoords
from imgProcessor.interpolate.polyfit2d import polyfit2dGrid
from camera.flatField.interpolationMethods import function






class DeviceAndGridSlice(object):
    '''
    TODO
    '''
    def __init__(self, n, p0,p1, n0,n1, g0,g1, n_img_row):
        self.n = n
        self.n0 = n0
        self.n1 = n1
        self.g0,self.g1 = g0,g1
        self.p0init = p0
        self.p1init = p1
        self.n_img_row = n_img_row
    
    def iter(self):
        c = 1
        p0,p1 = self.p0init,self.p1init
        
        for i in range(self.n):
            bottom = -1<p0     # bottom of device is in grid
            top = p0+self.n0<=self.g0    # top of device in grid
            left = -1<p1       # left of device in grid
            right = p1+self.n1<=self.g1  # right of device in grid

            if left and right:
                q1 =  slice(None, self.n1)
                qq1 = slice(p1  , p1+self.n1)
            elif not left and right:
                q1 =  slice(-p1 ,self.n1)
                qq1 = slice(None,p1+self.n1)
            elif not right and left:
                q1 =  slice(None,-(p1+self.n1-self.g1))
                qq1 = slice(p1  ,None)        
            else:
                q1 =  slice(-p1,-(p1+self.n1-self.g1))
                qq1 = slice(p1  ,p1+self.n1)
                     
            if bottom and top:
                q0 =  slice(None,self.n0)
                qq0 = slice(p0  , p0+self.n0)
            elif not bottom and top:
                q0 =  slice(-p0 ,self.n0)
                qq0 = slice(None,p0+self.n0)
            elif not top and bottom:
                q0 =  slice(None,-(p0+self.n0-self.g0))
                qq0 = slice(p0  ,None)
            else:
                q0 =  slice(-p0,-(p0+self.n0-self.g0))
                qq0 = slice(None  ,p0+self.n0)
            yield i, q0,q1,qq0,qq1

            
            p0 += 1
            if c == self.n_img_row:
                p0 = self.p0init
                p1 += 1
                c = 0
            c+=1




def flatFieldDiscreteSteps(imgs, n_cells_device, initial_cell_position,
                           n_img_row,#n_steps, 
                           d_step,
                           initial_px_position=(0,0),
                           bg_img=None,
                           bg_value=0, main_direction='y',
                           fit_fn='polynomial',
                           visualize_fn=None,
                           max_iter=100,max_dev=1e-6):
    '''
    imgs --> either image(-path) of all images in right order
    n_cells_device(x,y) --> how many cells does the device have
    initial_cell_position(x,y) --> position of the device in first image
        example for a 2x3 device
        (0,0) the device if completely in the edge of the first image
                    [1,1,0,0,0
                     1,1,0,0,0
                     1,1,0,0,0,
                     0,0,0,0,0]
        (0,-1) 1 cell row of the device is not in the image
                    [1,1,0,0,0
                     1,1,0,0,0
                     0,0,0,0,0,
                     0,0,0,0,0]   
        (1,0) the device is not at the edge but one column away
                    [0,1,1,0,0
                     0,1,1,0,0
                     0,1,1,0,0,
                     0,0,0,0,0]    
    n_img_row -> how many images are taken until the device jumps to the next column
    n_steps(x,y) -> how many different cells positions are in the image
    d_step(dx,dy) --> the pixel distance from in x,y
    
    OPTIONAL:
              
    initial_px_position(x,y) --> pixel position of bottom left in the image
        if device aligns perfectly with image this value is (0,0)       
    bg_value -> in order to exclude background when averaging
                set this value to > 0
                if bg_img is given, bg_value should be a threshold between background and signal
    
    main_direction ('x' OR 'y') --> first direction the the positional 
                change of the device
    '''
    
    
    assert main_direction=='y', 'can only move in y at the moment'
    
    if bg_img is not None:
        bg_img = imread(bg_img)

    d1,d0 = d_step
    n1,n0 = n_cells_device
    f1,f0 = initial_px_position
    p1init,p0init = initial_cell_position
    
    imgs[0] = imread(imgs[0])
    s0,s1 = imgs[0].shape[:2]
    
    mirror_y,mirror_x = False, False
    if d0 < 0:
        mirror_y = True
        d0 *= -1
        f0 = s0-f0
    if d1 < 0:
        mirror_x = True
        d1 *= -1
        f1 = s1-f1

    #grid resolution:
    g0,g1 = int(s0/d0)+1,  int(s1/d1)+1
    devices = np.full((len(imgs), n0, n1), np.nan)
    grid = np.full((len(imgs), g0,g1), np.nan)

    def fn(x):
        #average excluding background
        ind = x>bg_value
        #majority is background:
        if ind.sum() < 0.5*ind.size:
            return np.nan
        return x[ind].mean()

    
    gen = DeviceAndGridSlice(len(imgs),p0init,p1init, n0,n1, g0,g1, n_img_row) 
    
    
    for (img, (n, q0,q1,qq0,qq1)) in zip(imgs,gen.iter()):
        img = imread(img,dtype=float)
        
        if bg_img is not None:
            img -= bg_img
        
        if mirror_y:
            img = img[::-1]
        if mirror_x:
            img = img[:,::-1]
       
        #average image to grid:  
        p01 = (f0-d0*p0init, f1-d1*p1init)   

        g = subCell2DFnArray( img,fn,(g0,g1),  d01=(d0,d1), p01=p01)#(f0,f1) )
        grid[n] = g
        #assign averaged cell values to device:  
        devices[n, q0, q1] = g[qq0,qq1]

        if visualize_fn:
            visualize_fn(grid[n], devices[n], img, 
                         subCell2DCoords(img, (g0,g1), 
                                         p01=p01, d01=(d0,d1)))

    #INITIAL FLAT FIELD:
    ff = np.nanmean(grid, axis=0)
    if bg_img is None:
        ff -= bg_value
    ff/=np.nanmax(ff)
    ffs = np.full_like(grid, np.nan)
    

    #OBJECT (flat field corrected):
    obj = np.full_like(devices, np.nan)
    n = 0    
    #SAPARATE OBJECT AND FLATFIELD THROUGH INTERATION:    
    while True:
        for (n, q0,q1,qq0,qq1) in gen.iter():
                obj[n,q0,q1] = devices[n,q0,q1]/ff[qq0,qq1]
        avgobj = np.nanmean(obj, axis=0)
        #NEXT FLATFIELD
        for (n, q0,q1,qq0,qq1) in gen.iter():
            ffs[n,qq0,qq1] =  devices[n,q0,q1] / avgobj[q0,q1]
        ff2 = np.nanmean(ffs, axis=0)

        dev = (np.nanmean((ff - ff2)**2))**0.5#RMS
        ff = ff2
        if n > max_iter or dev < max_dev:
            break
        n+=1
 
    
    pointdensity = len(imgs)-np.sum(np.isnan(grid), axis=0)
    
    #there are pixels missing if grid doesn't go throughout full image:
    left = p01[1]
    bottom = p01[0]
    top = bottom + (d0*g0)
    right = left + (d1*g1) 

    #TODO: maybe put meshgrid into own function
    #build meshgrid for rescaling:
    px = np.r_[#first row 
               np.linspace(bottom, bottom+d0,d0+bottom),
               #middle part
               np.arange(bottom + d0, bottom + d0*(g0-1), dtype=float),
               #last row
               np.linspace(top-d0,s0,s0-(top-d0))]
 
    py = np.r_[#first column
               np.linspace(left, left+d1,d1+left),
               #middle part
               np.arange(left + d1, left + d1*(g1-1), dtype=float),
               #last column
               np.linspace(right-d1,s1,s1-(right-d1))]

    xx,yy = np.meshgrid(py,px)
    yy= yy / s0 * (g0-1)
    xx= xx /s1 * (g1-1)

    #replace nan and recale:
    if fit_fn == 'polynomial':
        ff2 = polyfit2dGrid(ff,np.isnan(ff),outgrid=(yy,xx))
    else:
        ff2 = function(ff, np.isnan(ff), outgrid=(yy,xx))
    return ff, ff2, avgobj, pointdensity



if __name__ == '__main__':
    pass
    #TODO: make test case

#     import pylab as plt
#     from fancytools.os.PathStr import PathStr   
#     p = PathStr()
#     bg_img = r''
#     initial_px_position = (-2,3350)#x,y
#     d_step = (219,-260)#x,y
#     n_steps = (15,10)
#     n_img_row = 10 #how many images in main direction till jump
#     bg_value = 200  
#     imgs = sorted(p.files()) 
#     main_direction=0
#     n_cells_device = (6,13)#x,y
#     initial_cell_position = 0,-3#x,y
# 
#     (grid, grid2, device, pointdensity) = flatFieldDiscreteSteps(
#             imgs, n_cells_device, 
#             initial_cell_position,
#             n_img_row,#n_steps, 
#             d_step,
#             initial_px_position,
#             bg_img=bg_img,
#             bg_value=bg_value, main_direction='y',
#             fit_fn='polynomial',#'function'
#             #visualize_fn=visualize
#             )
# 
#     plt.figure('device')
#     plt.imshow(device, interpolation='none')
#     plt.figure('pointdensity')
#     plt.imshow(pointdensity, interpolation='none')
#     plt.colorbar()
#     plt.figure('raw flat field')
#     plt.imshow(grid, interpolation='none')
#     plt.figure('interpolated flat field')
#     plt.imshow(grid2, interpolation='none')
#     plt.colorbar()
#     plt.show()


#     