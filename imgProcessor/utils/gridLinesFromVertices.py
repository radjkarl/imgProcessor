# coding=utf-8
from __future__ import print_function

import numpy as np
import cv2
from imgProcessor.utils.sortCorners import sortCorners

#
# def gridLinesFromVertices_old(edges, nCells, dtype=float):
#     """creates a regular 2d grid from given edge points (4*(x0,y0))
#     and number of cells in x and y
#
#     Returns:
#         tuple(4lists): horizontal and vertical lines as (x0,y0,x1,y1)
#     """
#     e = edges
#     sx, sy = nCells[0] + 1, nCells[1] + 1
#     # horizontal lines
#     x0 = np.linspace(e[0, 0], e[3, 0], sy, dtype=dtype)
#     x1 = np.linspace(e[1, 0], e[2, 0], sy, dtype=dtype)
#
#     y0 = np.linspace(e[0, 1], e[3, 1], sy, dtype=dtype)
#     y1 = np.linspace(e[1, 1], e[2, 1], sy, dtype=dtype)
#
#     horiz = np.array(list(zip(x0, y0, x1, y1)))
#
#     # vertical lines
#     x0 = np.linspace(e[0, 0], e[1, 0], sx, dtype=dtype)
#     x1 = np.linspace(e[3, 0], e[2, 0], sx, dtype=dtype)
#
#     y0 = np.linspace(e[0, 1], e[1, 1], sx, dtype=dtype)
#     y1 = np.linspace(e[3, 1], e[2, 1], sx, dtype=dtype)
#
#     vert = np.array(list(zip(x0, y0, x1, y1)))
#
#     return horiz, vert


def gridLinesFromVertices(edges, nCells, subgrid=None, dtype=float):
    """
    ###TODO  REDO TXT

    OPTIONAL:
    subgrid = ([x],[y]) --> relative positions
        e.g. subgrid = ( (0.3,0.7), () )
             --> two subgrid lines in x - nothing in y

    Returns: 
        horiz,vert -> arrays of (x,y) poly-lines


    if subgrid != None, Returns:
            horiz,vert, subhoriz, subvert


    #######
    creates a regular 2d grid from given edge points (4*(x0,y0))
    and number of cells in x and y

    Returns:
        tuple(4lists): horizontal and vertical lines as (x0,y0,x1,y1)
    """

    nx, ny = nCells

    y, x = np.mgrid[0.:ny + 1, 0.:nx + 1]

    src = np.float32([[0, 0], [nx, 0], [nx, ny], [0, ny]])
    dst = sortCorners(edges).astype(np.float32)

    homography = cv2.getPerspectiveTransform(src, dst)

    pts = np.float32((x.flatten(), y.flatten())).T
    pts = pts.reshape(1, *pts.shape)

    pts2 = cv2.perspectiveTransform(pts, homography)[0]

    horiz = pts2.reshape(ny + 1, nx + 1, 2)
    vert = np.swapaxes(horiz, 0, 1)

    subh, subv = [], []
    if subgrid is not None:
        sh, sv = subgrid

        if len(sh):
            subh = np.empty(shape=(ny * len(sh), nx + 1, 2), dtype=np.float32)
            last_si = 0
            for n, si in enumerate(sh):
                spts = pts[:, :-(nx + 1)]
                spts[..., 1] += si - last_si
                last_si = si
                spts2 = cv2.perspectiveTransform(spts, homography)[0]
                subh[n::len(sh)] = spts2.reshape(ny, nx + 1, 2)
        if len(sv):
            subv = np.empty(shape=(ny + 1, nx * len(sv), 2), dtype=np.float32)
            last_si = 0
            sspts = pts.reshape(1, ny + 1, nx + 1, 2)
            sspts = sspts[:, :, :-1]

            sspts = sspts.reshape(1, (ny + 1) * nx, 2)
            for n, si in enumerate(sv):
                sspts[..., 0] += si - last_si
                last_si = si
                spts2 = cv2.perspectiveTransform(sspts, homography)[0]
                subv[:, n::len(sv)] = spts2.reshape(ny + 1, nx, 2)
            subv = np.swapaxes(subv, 0, 1)
    return [horiz, vert, subh, subv]


if __name__ == '__main__':
    import sys
    import pylab as plt

    edges = np.array([(0, 0),
                      (1, 0.1),
                      (2, 2),
                      (0.1, 1)])
    ncells = (10, 5)

    out = []
    # case1: simple grid
    out.append(gridLinesFromVertices(edges, ncells))
    # case 2: 2 horizontal sublines
    out.append(
        gridLinesFromVertices(edges, ncells, subgrid=((0.3, 0.7), None)))
    # case 3: 2 vertical sublines
    out.append(
        gridLinesFromVertices(edges, ncells, subgrid=(None, (0.3, 0.7))))

    if 'no_window' not in sys.argv:
        f, ax = plt.subplots(3)
        for c, title, aa in zip(out, (
                "simple grid with %s cells" % str(ncells),
                "same with 2 horizontal sublines",
                "same with 2 vertical sublines"), ax):
            aa.set_title(title)

            h, v, sh, sv = c

            for polyline in h:
                aa.plot(polyline[:, 0], polyline[:, 1], 'o-', color='r')
            for polyline in sh:
                aa.plot(polyline[:, 0], polyline[:, 1], 'o-', color='b')
            for polyline in v:
                aa.plot(polyline[:, 0], polyline[:, 1], 'o-', color='g')

            aa.scatter(edges[:, 0], edges[:, 1], s=100)

        plt.show()


#     #old
#     h, v = gridLinesFromVertices(edges, ncells)
#     print(h[0])
#
#     if 'no_window' not in sys.argv:
#         plt.figure(
#             'create grid with %s cells within given edge points' %
#             str(ncells))
#         for l in v:
#             plt.plot((l[0], l[2]), (l[1], l[3]), 'r')
#         for l in h:
#             plt.plot((l[0], l[2]), (l[1], l[3]), 'g')
#
#         plt.scatter(edges[:, 0], edges[:, 1])
#         plt.show()
