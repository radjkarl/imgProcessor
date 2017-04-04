# coding=utf-8
from __future__ import division

from imgProcessor.imgIO import imread


def imgAverage(images, copy=True):
    '''
    returns an image average

    works on many, also unloaded images
    minimises RAM usage
    '''
    i0 = images[0]
    out = imread(i0, dtype='float')
    if copy and id(i0) == id(out):
        out = out.copy()

    for i in images[1:]:
        out += imread(i, dtype='float')
    out /= len(images)
    return out
