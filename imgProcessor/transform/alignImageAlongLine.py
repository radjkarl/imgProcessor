# coding=utf-8
from __future__ import division

import cv2
import numpy as np

import fancytools.math.line as ln


def alignImageAlongLine(img, line, height=15, length=None,
                        zoom=1, fast=False, borderValue=0):
    '''
    return a sub image aligned along given line

    @param img -  numpy.2darray input image to get subimage from
    @param line - list of 2 points [x0,y0,x1,y1])
    @param height - height of output array in y
    @param length - width of output array
    @param zoom - zoom factor
    @param fast - speed up calculation using nearest neighbour interpolation
    @returns transformed image as numpy.2darray with found line as in the middle
    '''

    height = int(round(height))
    if height % 2 == 0:  # ->is even number
        height += 1  # only take uneven numbers to have line in middle
    if length is None:
        length = int(round(ln.length(line)))
    hh = (height - 1)
    ll = (length - 1)

    # end points of the line:
    p0 = np.array(line[0:2], dtype=float)
    p1 = np.array(line[2:], dtype=float)
    # p2 is above middle of p0,p1:
    norm = np.array(ln.normal(line))
    if not ln.isHoriz(line):
        norm *= -1

    p2 = (p0 + p1) * 0.5 + norm * hh * 0.5
    middleY = hh / 2
    pp0 = [0, middleY]
    pp1 = [ll, middleY]
    pp2 = [ll * 0.5, hh]

    pts1 = np.array([p0, p1, p2], dtype=np.float32)
    pts2 = np.array([pp0, pp1, pp2], dtype=np.float32)

    if zoom != 1:
        length = int(round(length * zoom))
        height = int(round(height * zoom))
        pts2 *= zoom

    # TRANSFORM:
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(
        img, M, (length, height),
        flags=cv2.INTER_NEAREST if fast else cv2.INTER_LINEAR,
        borderValue=borderValue)
    return dst


if __name__ == '__main__':
    import sys
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    from imgProcessor.imgIO import imread

    img = imread(PathStr(imgProcessor.__file__).dirname().join(
        'media', 'electroluminescence', 'EL_module_orig.PNG'))

    line = (48, 325, 162, 54)  # x0,y0,x1,y1
#     line = (0,200,150,200)

    sub = alignImageAlongLine(img, line, height=40)

    if 'no_window' not in sys.argv:

        plt.figure('original')
        plt.imshow(img)
        plt.plot((line[0], line[2]), (line[1], line[0o3]))

        plt.figure('sub image')
        plt.imshow(sub)

        plt.show()

        # TEST angular dependency:
        s = 500, 500
        m = 250, 250
        lenline = 200
        n = 5
        h = 55
        img = np.zeros((s[0], s[1]))

        # draw few lines through center:
        lines = []
        d = 0
        dd = np.pi / n
        c = 10
        dc = (255. - c) / n
        for i in range(n):
            d += dd
            p0 = int(-lenline * np.cos(d) +
                     m[0]), int(-lenline * np.sin(d) + m[1])
            p1 = int(lenline * np.cos(d) + m[0]
                     ), int(lenline * np.sin(d) + m[1])
            cv2.line(img, p0, p1, color=int(c))
            c += dc
            lines.append((p0[0], p0[1], p1[0], p1[1]))

        plt.figure(0)
        fig = plt.imshow(img, interpolation='none')
        clim = fig.get_clim()
        for i, sub in enumerate(plt.subplots(2)[1]):

            sub.set_title(str(lines[i]))
            fig = sub.imshow(alignImageAlongLine(
                img, lines[i], height=h, fast=True),
                clim=clim, interpolation='none')
            sub.set_aspect('auto')

        plt.show()
