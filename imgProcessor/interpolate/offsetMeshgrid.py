from __future__ import division

import numpy as np

def offsetMeshgrid(offset, grid, shape):
    '''
    Imagine you have cell averages [grid] on an image.
    the top-left position of [grid] within the image 
    can be variable [offset]
    
    offset(x,y) 
        e.g.(0,0) if no offset
    grid(nx,ny) resolution of smaller grid
    shape(x,y) -> output shape 
    
    returns meshgrid to be used to upscale [grid] to [shape] resolution
    '''    
    g0,g1 = grid
    s0,s1 = shape
    o0, o1 = offset
    #rescale to small grid:
    o0 = - o0/ s0 * (g0-1)
    o1 = - o1/ s1 * (g1-1)

    xx,yy = np.meshgrid(np.linspace(o1, o1+g1-1, s1),
                        np.linspace(o0,o0+g0-1,  s0))
    return yy,xx


#     i0 = np.arange(-bottom, d0)#,d0+bottom+1)#first row 
#     i1 = np.arange(d0+i0[1]-i0[0], bottom + d0*(g0-1), dtype=float)#middle part
#     i2 = np.linspace(i1[-1],s0,s0-(top-d0)+2)[1:]##last row
#     i2 = np.arange(i1[-1],top,)[1:]##last row
# 
#     py = np.r_[i0,i1,i2]
#  
#     i0 = np.linspace(-left, d1,d1+left+1)#first column
#     i1 = np.arange(d1+i0[1]-i0[0], left + d1*(g1-1), dtype=float)#middle part
#     i2 =  np.linspace(i1[-1],s1, s1-(right-d1)+2)[1:]#last column
#     px = np.r_[i0,i1,i2]
# 
#     xx,yy = np.meshgrid(px,py)
#     yy= yy / s0 * (g0-1)
#     xx= xx /s1 * (g1-1)
#     return yy,xx



if __name__ == '__main__':
    #this example shows extrapolation on an offset grid
    #where first and last row/column are skewed depending 
    #on the size of [offs]
    
    import pylab as plt
    import sys
    from imgProcessor.interpolate.polyfit2d import polyfit2dGrid
    
    g = (7,10)#small grid size
    small = np.fromfunction(lambda x,y: np.sin(7*x/g[0])
                            +np.cos(9*y/g[1]), g)
    g2 = (70,100) #big grid size

    if 'no_window' not in sys.argv:
        f, ax = plt.subplots(3,2)

    for i, offs in zip( (0,1,2), ((0,0),(8,9),(-5,-3)) ):
        ########
        yy,xx = offsetMeshgrid(offs, g, g2)
        big = polyfit2dGrid(small, outgrid=(yy,xx),
                            order=7)
        #######
        if 'no_window' not in sys.argv:
            ax[i,0].set_title('input data')
            ax[i,0].imshow(small, interpolation='none')
            

            ax[i,1].set_title('Rescaled with offset grid {}'.format(offs))    
            ax[i,1].imshow(big, interpolation='none')

    if 'no_window' not in sys.argv:
        plt.show()