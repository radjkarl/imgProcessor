from __future__ import division
import numpy as np


def guessVignettingParam(shape):
    return (shape[0] * 0.7, 0, 0, 0, shape[0] / 2, shape[1] / 2)


def vignetting(xy, f=100, alpha=0, rot=0, tilt=0, cx=50, cy=50):
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
    x, y = xy
    # distance to image center:
    dist = ((x - cx)**2 + (y - cy)**2)**0.5

    # OFF_AXIS ILLUMINATION FACTOR:
    A = 1.0 / (1 + (dist / f)**2)**2
    # GEOMETRIC FACTOR:
    if alpha != 0:
        G = (1 - alpha * dist)
    else:
        G = 1
    # TILT FACTOR:
    if tilt != 0:
        T = tiltFactor((x, y), f, tilt, rot, (cy, cx))
    else:
        T = 1
    return A * G * T


def tiltFactor(xy, f, tilt, rot, center=None):
    '''
    this function is extra to only cover vignetting through perspective distortion

    f - focal length [px]
    tau - tilt angle of a planar scene [radian]
    rot - rotation angle of a planar scene [radian]
    '''
    x, y = xy
    arr = np.cos(tilt) * (
        1 + (np.tan(tilt) / f) * (
            x * np.sin(rot) - y * np.cos(rot)))**3
    return arr


if __name__ == '__main__':
    import pylab as plt
    import sys

    param = {'cx': 75,
             'cy': 50,
             'tilt': 0.2,
             'rot': 0.3}
    vig = np.fromfunction(lambda y, x: vignetting((x, y), **param), (100, 150))

    param = {'f': 150,
             'rot': 0,
             'tilt': np.radians(60)}
    tilt = np.fromfunction(lambda y, x: tiltFactor((x, y), **param), (75, 75))

    if 'no_window' not in sys.argv:
        plt.figure('vignetting')
        plt.imshow(vig)
        plt.colorbar()

        plt.figure('tilt factor only')
        plt.imshow(tilt)
        plt.colorbar()

        plt.show()
