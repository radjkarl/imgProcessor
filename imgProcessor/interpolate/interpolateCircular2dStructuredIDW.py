import numpy as np
from numba import jit
from math import atan2

@jit(nopython=True)
def interpolateCircular2dStructuredIDW(grid, mask, kernel=15, power=2, 
                               fr=1, fphi=1, cx=0, cy=0):
    '''
    same as interpolate2dStructuredIDW
    but calculation distance to neighbour using polar coordinates
    fr, fphi --> weight factors for radian and radius differences
    cx,cy -> polar center of the array e.g. middle->(sx//2+1,sy//2+1) 
    '''
    gx = grid.shape[0]
    gy = grid.shape[0]

    #FOR EVERY PIXEL
    for i in xrange(gx):
        for j in xrange(gy):
            
            if mask[i,j]:
                
                xmn = i-kernel
                if xmn < 0:
                    xmn = 0
                xmx = i+kernel
                if xmx > gx:
                    xmx = gx
                    
                ymn = j-kernel
                if ymn < 0:
                    ymn = 0
                ymx = j+kernel
                if ymx > gx:
                    ymx = gy
                    
                sumWi = 0.0
                value = 0.0 

                #radius and radian to polar center:
                R = ((i-cx)**2+(j-cy)**2)**0.5
                PHI = atan2(j-cy, i-cx)
                
                #FOR EVERY NEIGHBOUR IN KERNEL              
                for xi in xrange(xmn,xmx):
                    for yi in xrange(ymn,ymx):
                        if  (xi != i or yi != j) and not mask[xi,yi]:
                            nR = ((xi-cx)**2+(yi-cy)**2)**0.5
                            dr =  R - nR
                            #average radius between both p:
                            midR = 0.5*(R+nR)
                            #radian of neighbour p:
                            nphi = atan2(yi-cy, xi-cx)
                            #relative angle between both points:
                            dphi = min((2*np.pi) - abs(PHI - nphi), 
                                       abs(PHI - nphi))    
                            dphi*=midR       
                            
                            dist = ((fr*dr)**2+(fphi*dphi)**2)**2

                            wi = 1.0 / dist**(0.5*power)
                            sumWi += wi
                            value += wi * grid[xi,yi]  
                if sumWi:
                    grid[i,j] = value/sumWi               

    return grid





if __name__ == '__main__':
    import sys
    from matplotlib import pyplot as plt
    
    #this is part or a point spread function
    arr = np.array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.00091412,  0.00092669,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.00071046,  0.00087626,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.00174763,  0.00316936,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.00802817,  0.01606653,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.00052719,  0.00165561,  0.00208777,  0.00212379,  0.        ,
         0.        ,  0.        ,  0.01836601,  0.04002059,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.00052719,  0.00257932,  0.00291309,  0.00914339,  0.02844799,
         0.04823197,  0.05040033,  0.06361089,  0.04638128,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.00225948,  0.00222627,  0.00755744,  0.0133372 ,
         0.02761284,  0.06116419,  0.07565894,  0.05202775,  0.01511698,
         0.00697312,  0.00270475,  0.00077251,  0.00067585,  0.00055524],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.0532791 ,  0.06633063,  0.07244685,  0.03513939,
         0.01519723,  0.00217622,  0.00107757,  0.00076782,  0.0004534 ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.04717624,  0.03095101,  0.        ,  0.        ,
         0.        ,  0.00164764,  0.00137625,  0.00075694,  0.00076486],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.02333552,  0.01279662,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.00623037,  0.00400915,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.00128086,  0.00131918,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.00080955,  0.00085656,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.00094004,  0.00078282,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
    mask = arr==0
    s = arr.shape
    cx = s[0]//2+1
    cy =  s[1]//2+1
    fit = interpolateCircular2dStructuredIDW(
        arr.copy(), mask, fr=1, fphi=0.2, cx=cx, cy=cy)

    if 'no_window' not in sys.argv:
        plt.figure('original')
        plt.imshow(arr, interpolation='none')
        plt.colorbar()
        
        plt.figure('fit')
        plt.imshow(fit, interpolation='none')
        plt.colorbar()
        
        plt.show()
    

    