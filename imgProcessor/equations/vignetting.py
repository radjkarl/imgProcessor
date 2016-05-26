import numpy as np


def guessVignettingParam(arr):
    return (arr.shape[0]*0.7, 0, 0, 0,arr.shape[0]/2.0,arr.shape[1]/2.0)


def vignetting((x, y), f=100, alpha=0, rot=0, tilt=0, cx=50, cy=50):
    '''
    Vignetting equation using the KANG-WEISS-MODEL
    see http://research.microsoft.com/en-us/um/people/sbkang/publications/eccv00.pdf   
     
    f - focal length
    alpha - coefficient in the geometric vignetting factor
    tilt - tilt angle of a planar scene
    rot - rotation angle of a planar scene
    cx - image center, x
    cy - image center, y
    '''
    #distance to image center:
    dist = ((x-cx)**2 + (y-cy)**2)**0.5
    
    #OFF_AXIS ILLUMINATION FACTOR:
    A = 1.0/(1+(dist/f)**2)**2
    #GEOMETRIC FACTOR:
    G = (1-alpha*dist)
    #TILT FACTOR:
    T = tiltFactor((x,y), f, tilt, rot)

    return A*G*T


def tiltFactor((x, y), f, tilt, rot):
    '''
    this function is extra to only cover vignetting through perspective distortion
    
    f - focal length
    tau - tilt angle of a planar scene
    Xi - rotation angle of a planar scene
    '''
    return np.cos(tilt) * (1+(np.tan(tilt)/f) * (x*np.sin(rot)-y*np.cos(rot)) )**3



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import sys
        
    param = {'cx':50, 
             'cy':50,
             'tilt':-0.5}
    vig = np.fromfunction(lambda x,y: vignetting((x,y), **param), (100,150))
    


    param = {'f':100,
             'rot':2,
             'tilt':0.1}
    tilt = np.fromfunction(lambda x,y: tiltFactor((x,y), **param), (100,150))

    if 'no_window' not in sys.argv:
        plt.figure('vignetting')
        plt.imshow(vig)
        plt.colorbar()

        plt.figure('tilt factor only')
        plt.imshow(tilt)
        plt.colorbar()
    
        plt.show()
    