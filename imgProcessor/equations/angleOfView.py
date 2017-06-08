import numpy as np

# 
# def _fn(XY, a, c0, c1):
#     x,y = XY
#     rx = (x-c0)**2
#     ry = (y-c1)**2  
#     return (1+np.tan(a)*((rx+ry)/c0))**0.5
#     


def angleOfView(XY, shape=None, a=None, f=None, D=None, center=None):
    '''
    Another vignetting equation from:
    M. Koentges, M. Siebert, and D. Hinken, "Quantitative analysis of PV-modules by electroluminescence images for quality control"
        2009
    f --> Focal length
    D --> Diameter of the aperture
        BOTH, D AND f NEED TO HAVE SAME UNIT [PX, mm ...]
    a --> Angular aperture
    
    center -> optical center [y,x]
    '''
    if a is None:
        assert f is not None and D is not None
        #https://en.wikipedia.org/wiki/Angular_aperture
        a = 2*np.arctan2(D/2,f)
    
    x,y = XY

    try:
        c0,c1 = center
    except:
        s0,s1 = shape
        c0,c1 = s0/2, s1/2

    rx = (x-c0)**2
    ry = (y-c1)**2  

    return  1 / (1+np.tan(a)*((rx+ry)/c0))**0.5
    

def angleOfView2(x,y, b, x0=None,y0=None):
    '''
    Corrected AngleOfView equation by Koentges (via mail from 14/02/2017)
    b --> distance between the camera and the module in m
    x0 --> viewable with in the module plane of the camera in m
    y0 --> viewable height in the module plane of the camera in m
    x,y --> pixel position [m] from top left
    '''
    if x0 is None:
        x0 = x[-1,-1]
    if y0 is None:
        y0 = y[-1,-1]    
    return np.cos( np.arctan( np.sqrt(
                    ( (x-x0/2)**2+(y-y0/2)**2 ) ) /b  ) )







if __name__ == '__main__':
    import pylab as plt
    import sys
    s = (600,800)
    
    f = 200
    D = 1
    
    arr= np.fromfunction(lambda x,y: angleOfView2(x,y, 2000), s)

#     arr= np.fromfunction(lambda x,y: angleOfView((x,y), s ,f=f, D=D), s)

    if 'no_window' not in sys.argv:
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
