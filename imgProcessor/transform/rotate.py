# coding=utf-8
from __future__ import division

import numpy as np
import cv2


def rotate(image, angle, interpolation=cv2.INTER_CUBIC,
           borderMode=cv2.BORDER_REFLECT, borderValue=0):
    '''
    angle [deg]
    '''
    s0, s1 = image.shape
    image_center = (s0 - 1) / 2., (s1 - 1) / 2.
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape,
                            flags=interpolation, borderMode=borderMode,
                            borderValue=borderValue)
    return result


if __name__ == '__main__':
    import pylab as plt
    import sys

    a = np.linspace(0, 1, 50)
    a = np.tile(a, (50, 1))

    b = rotate(a, 14)
    c = rotate(b, -14)

    assert np.abs(a - c).mean() < 0.005

    if 'no_window' not in sys.argv:

        plt.figure('in')
        plt.imshow(a)
        plt.figure('rotated')
        plt.imshow(b)
        plt.figure('back rotated')
        plt.imshow(c)

        plt.show()
