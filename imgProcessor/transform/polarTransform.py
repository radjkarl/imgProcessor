from __future__ import division

# origin [now heavily modified]
# http://stackoverflow.com/questions/9924135/fast-cartesian-to-polar-to-cartesian-in-python
# https://imagej.nih.gov/ij/plugins/download/Polar_Transformer.java

import numpy as np
import cv2


def _polar2cart(r, phi, center):
    x = r * np.cos(phi) + center[0]
    y = r * np.sin(phi) + center[1]
    return x, y


def _cart2polar(x, y, center):
    xx = x - center[0]
    yy = y - center[1]
    phi = np.arctan2(yy, xx)
    r = np.hypot(xx, yy)
    return r, phi


def linearToPolarMaps(shape, center=None, final_radius=None,
                      initial_radius=None, phase_width=None):
    s0, s1 = shape
    if center is None:
        center = (s0 - 1) / 2, (s1 - 1) / 2
    if final_radius is None:
        final_radius = ((0.5 * s0)**2
                        + (0.5 * s1)**2)**0.5
    if initial_radius is None:
        initial_radius = 0
    if phase_width is None:
        phase_width = 2 * np.pi * final_radius

    phi, R = np.meshgrid(np.linspace(1.5 * np.pi, -0.5 * np.pi, phase_width),
                         np.arange(initial_radius, final_radius))

    mapX, mapY = _polar2cart(R, phi, center)
    return mapY.astype(np.float32), mapX.astype(np.float32)


def linearToPolar(img, center=None,
                  final_radius=None,
                  initial_radius=None,
                  phase_width=None,
                  interpolation=cv2.INTER_AREA, maps=None,
                  borderValue=0, borderMode=cv2.BORDER_REFLECT, **opts):
    '''
    map a 2d (x,y) Cartesian array to a polar (r, phi) array
    using opencv.remap
    '''
    if maps is None:
        mapY, mapX = linearToPolarMaps(img.shape[:2], center, final_radius,
                                       initial_radius, phase_width)
    else:
        mapY, mapX = maps

    o = {'interpolation': interpolation,
         'borderValue': borderValue,
         'borderMode': borderMode}
    o.update(opts)

    return cv2.remap(img, mapY, mapX, **o)


def polarToLinearMaps(orig_shape, out_shape=None, center=None):
    s0, s1 = orig_shape
    if out_shape is None:
        out_shape = (int(round(2 * s0 / 2**0.5)) - (1 - s0 % 2),
                     int(round(2 * s1 / (2 * np.pi) / 2**0.5)))
    ss0, ss1 = out_shape

    if center is None:
        center = ss1 // 2, ss0 // 2

    yy, xx = np.mgrid[0:ss0:1., 0:ss1:1.]
    r, phi = _cart2polar(xx, yy, center)
    # scale-pi...pi->0...s1:
    phi = (phi + np.pi) / (2 * np.pi) * (s1 - 2)

    return phi.astype(np.float32), r.astype(np.float32)


def polarToLinear(img, shape=None, center=None, maps=None,
                  interpolation=cv2.INTER_AREA,
                  borderValue=0, borderMode=cv2.BORDER_REFLECT, **opts):
    '''
    map a 2d polar (r, phi) polar array to a  Cartesian (x,y) array
    using opencv.remap
    '''

    if maps is None:
        mapY, mapX = polarToLinearMaps(img.shape[:2], shape, center)
    else:
        mapY, mapX = maps

    o = {'interpolation': interpolation,
         'borderValue': borderValue,
         'borderMode': borderMode}
    o.update(opts)

    return cv2.remap(img, mapY, mapX, **o)


if __name__ == '__main__':
    import sys
    import pylab as plt
    from skimage.transform import resize

    # CREATE a array filled with circles of random colour:
    a1 = np.ndarray((513, 513), dtype=np.float32)
    for i in range(10, 600, 10):
        cv2.circle(a1, (256, 256), i - 10,
                   np.random.randint(0, 255), thickness=4)
    a1 /= a1.max()
    a1 = resize(a1, (513, 255))

    # MAP to polar:
    a2 = linearToPolar(a1, interpolation=cv2.INTER_NEAREST)
    # MAP BACK to cartesian
    a3 = polarToLinear(a2, interpolation=cv2.INTER_NEAREST, shape=a1.shape)

    # ERROR between orig. and 2x transformed needs to be smaller than 13
    assert np.abs(a1 - a3).mean() < 13

    # PLOT
    if 'no_window' not in sys.argv:
        plt.figure('1. original')
        plt.imshow(a1, interpolation='none')
        cc = plt.colorbar().get_clim()

        plt.figure('2. polar map')
        plt.imshow(a2, interpolation='none')

        fig = plt.figure('3. mapped back to Cartesian')
        plt.imshow(a3, interpolation='none')
        plt.colorbar().set_clim(*cc)

        plt.show()
