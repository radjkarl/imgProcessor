import numpy as np

# 
# def _fn(XY, a, c0, c1):
#     x,y = XY
#     rx = (x-c0)**2
#     ry = (y-c1)**2  
#     return (1+np.tan(a)*((rx+ry)/c0))**0.5
#     

def angleOfView(XY, shape, a=None, f=None, D=None, center=None):
    '''
    Another vignetting equation from:
    M. Koentges, M. Siebert, and D. Hinken, "Quantitative analysis of PV-modules by electroluminescence images for quality control"
    
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

    s0,s1 = shape
    try:
        c0,c1 = center
    except:
        c0,c1 = s0/2, s1/2

    rx = (x-c0)**2
    ry = (y-c1)**2  
#     print (a, c0,c1)
#     import pylab as plt
#     arr = 1 / (1+np.tan(a)*((rx+ry)/c0))**0.5
#     
#     plt.imshow(arr.reshape(shape))
#     plt.show()

    return  1 / (1+np.tan(a)*((rx+ry)/c0))**0.5
    
        

if __name__ == '__main__':
    import pylab as plt
    import sys
    s = (600,800)
    
    f = 200
    D = 1
    

    arr= np.fromfunction(lambda x,y: angleOfView((x,y), s ,f=f, D=D), s)

    if 'no_window' not in sys.argv:
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
