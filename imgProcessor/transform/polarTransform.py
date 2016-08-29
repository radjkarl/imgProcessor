#origin [now heavily modified]
#http://stackoverflow.com/questions/9924135/fast-cartesian-to-polar-to-cartesian-in-python
#https://imagej.nih.gov/ij/plugins/download/Polar_Transformer.java

import numpy as np
import cv2


def _polar2cart(r, phi, center):
    x = r  * np.cos(phi) + center[0]
    y = r  * np.sin(phi) + center[1]
    return x, y


def _cart2polar(x, y,center):
    xx = x-center[0]
    yy = y-center[1]
    phi = np.arctan2(yy, xx)
    r = np.hypot(xx, yy)
    return r,phi


def linearToPolar(img, center=None, 
                  final_radius=None, 
                  initial_radius=None, 
                  phase_width=None, 
                  interpolation=cv2.INTER_CUBIC,
                  borderValue=0,borderMode=cv2.BORDER_REFLECT, **opts):
    s0,s1 = img.shape[:2]
    
    if center is None:
        center = s0/2.,s1/2.
    if final_radius is None:
        final_radius = ( (0.5*s0)**2
                        +(0.5*s1)**2)**0.5
    if initial_radius is None:
        initial_radius = 0  
    if phase_width is None:
        phase_width = 2*np.pi*final_radius

    phi , R = np.meshgrid(np.linspace(1.5*np.pi,-0.5*np.pi, phase_width), 
                          np.arange(initial_radius, final_radius))

    Xcart, Ycart = _polar2cart(R, phi, center)

    o = {'interpolation':interpolation, 
         'borderValue':borderValue, 
         'borderMode':borderMode}
    o.update(opts)

    return cv2.remap(img, Ycart.astype(np.float32), 
                          Xcart.astype(np.float32), **o )


def polarToLinear(img, shape=None, center=None,
                  interpolation=cv2.INTER_CUBIC,
                  borderValue=0, borderMode=cv2.BORDER_REFLECT, **opts):
    
    s0,s1 = img.shape[:2]
    
    if shape is None:
        shape =  ( int(round(s0*2 / 2**0.5)), 
                   int(round(2*s1/(2*np.pi) / 2**0.5)) )
    ss0,ss1 = shape
    
    if center is None:
        center= ss0/2, ss1/2
    
    yy,xx = np.mgrid[0:ss0:1., 0:ss1:1.]
    r, phi = _cart2polar(xx,yy, center)
    #scale-pi...pi->0...s1:
    phi = (phi+np.pi)/(2*np.pi) * (s1 -2)
    
    o = {'interpolation':interpolation, 
         'borderValue':borderValue,
         'borderMode':borderMode}
    o.update(opts)
    
    return cv2.remap(img, phi.astype(np.float32), 
                            r.astype(np.float32), **o)



if __name__ == '__main__':
    import pylab as plt
    
    #create a arrey filled with circles:
    a1 = np.ndarray((512,512),dtype=np.float32)
    for i in range(10,600,10): 
        cv2.circle(a1,(256,256),i-10,
                   np.random.randint(0,255),thickness=4)
    #map to polar:
    a2 = linearToPolar(a1)
    #map to cartesian
    a3 = polarToLinear(a2)
    
    #plot:
    plt.figure(1)
    plt.imshow(a1, interpolation='none')
    cc = plt.colorbar().get_clim()
    
    plt.figure(2)
    plt.imshow(a2, interpolation='none')    


    fig = plt.figure(3)
    plt.imshow(a3, interpolation='none')
    plt.colorbar().set_clim(*cc)
    
    plt.show()
