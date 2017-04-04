from __future__ import division

import numpy as np
import cv2

from imgProcessor.imgIO import imread
from imgProcessor.array.subCell2D import subCell2DFnArray, subCell2DCoords
from imgProcessor.interpolate.polyfit2d import polyfit2dGrid
from imgProcessor.camera.flatField.interpolationMethods import function
from imgProcessor.interpolate.offsetMeshgrid import offsetMeshgrid
from imgProcessor.equations.angleOfView import angleOfView


class _DeviceAndGridSlice(object):
    '''
    TODO
    '''

    def __init__(self, n, n0, n1, g0, g1, cell_positions):
        self.n = n
        self.n0 = n0
        self.n1 = n1
        self.g0, self.g1 = g0, g1
        self.cell_positions = cell_positions

    def iter(self):
        for i in range(self.n):

            p0, p1 = self.cell_positions[i]

            bottom = -1 < p0     # bottom of device is in grid
            top = p0 + self.n0 <= self.g0    # top of device in grid
            left = -1 < p1       # left of device in grid
            right = p1 + self.n1 <= self.g1  # right of device in grid
            if left and right:
                q1 = slice(None, self.n1)
                qq1 = slice(p1, p1 + self.n1)
            elif not left and right:
                q1 = slice(-p1, self.n1)
                qq1 = slice(None, p1 + self.n1)
            elif not right and left:
                q1 = slice(None, -(p1 + self.n1 - self.g1))
                qq1 = slice(p1, None)
            else:
                q1 = slice(-p1, -(p1 + self.n1 - self.g1))
                qq1 = slice(p1, p1 + self.n1)

            if bottom and top:
                q0 = slice(None, self.n0)
                qq0 = slice(p0, p0 + self.n0)
            elif not bottom and top:
                q0 = slice(-p0, self.n0)
                qq0 = slice(None, p0 + self.n0)
            elif not top and bottom:
                q0 = slice(None, -(p0 + self.n0 - self.g0))
                qq0 = slice(p0, None)
            else:
                q0 = slice(-p0, -(p0 + self.n0 - self.g0))
                qq0 = slice(None, p0 + self.n0)
            yield i, q0, q1, qq0, qq1


def positionsFromRaster(initial_cell_position, nimgs, nimgrow, cell_size,
                        initial_px_position,
                        direction, first_axis):
    '''
    input function for argument [positions] in vignettingDiscreteSteps()
    to be used, if device positions changes like a raster first
    monotoniously in one direction, than in another, like
    (0,0),(1,0),(2,0),(0,1),(1,1)...

    ###
    initial_cell_position(x,y) --> position of the device in first image
        example for a 2x3 device
        (0,0) the device if completely in the edge of the first image
                    [1,1,0,0,0
                     1,1,0,0,0
                     1,1,0,0,0,
                     0,0,0,0,0]
        (0,-1) 1 cell row of the device is not in the image
                    [1,1,0,0,0
                     1,1,0,0,0
                     0,0,0,0,0,
                     0,0,0,0,0]   
        (1,0) the device is not at the edge but one column away
                    [0,1,1,0,0
                     0,1,1,0,0
                     0,1,1,0,0,
                     0,0,0,0,0]  
    initial_px_position(x,y) --> pixel position of bottom left in the image
        if device aligns perfectly with image this value is (0,0)     
    n_img_row -> how many images are taken until the device jumps to the next column

    cell_size(x,y) --> number of pixels of one cell/step in x and y
    direction (0,1)--> first direction the the positional 
                          change of the device
    first_axis(0,1) -> whether major direction is in y(0) or x(1)

    '''
    assert first_axis in (0, 1)
    assert direction in (-1, 1)

    p1, p0 = p1init, p0init = initial_cell_position
    cell_positions = []

    c = 1
    for _ in range(nimgs):
        cell_positions.append((p0, p1))
        if first_axis == 0:
            p0 += direction
            if c == nimgrow:
                p0 = p0init
                p1 += 1
                c = 0
        else:
            p1 += direction
            if c == nimgrow:
                p1 = p1init
                p0 += 1
                c = 0
        c += 1

    f1, f0 = initial_px_position
    d1, d0 = cell_size
    return (f0, f1), cell_positions, (d0, d1)


def positionsFromTopLeftCorner(pos, cell_size):
    '''
    input function for argument [positions] in vignettingDiscreteSteps()
    to be used, if corner positions [pixels] of the device are known
    ###
    pos ((x,y),...) -> all pixel positions of the TOP LEFT corner of the device
    cell_size(x,y) --> number of pixels of one cell/step in x and y

    '''
    d1, d0 = cell_size

    f1, f0 = pos[0]
    # calculate cell positions
    p1init = int(f1 / d1) + 1 if f1 > 0 else 0

    p0init = int(f0 / d0) + 1 if f0 > 0 else 0
    ref1 = f1 - p1init * d1
    ref0 = f0 - p0init * d0

    cell_positions = [(p0init, p1init)]
    for x, y in pos[1:]:
        cell_positions.append((int(round((y - ref0) / d0)),
                               int(round((x - ref1) / d1))))

    return (f0, f1), cell_positions, (d0, d1)


def vignettingFromDiscreteSteps(imgs, n_cells_device,
                                positions,
                                bg_img=None,
                                bg_value=0, thresh=None,
                                postprocessing_method='polynomial',
                                visualize=False,
                                max_iter=100, max_dev=1e-6):
    '''
    This method is referred as 'Method D' in 

    ---
    K.Bedrich, M.Bokalic et al.:
    ELECTROLUMINESCENCE IMAGING OF PV DEVICES:
    ADVANCED FLAT FIELD CALIBRATION,2017
    ---

    imgs --> either images or image path of all images in right order
    n_cells_device(x,y) --> how many cells does the device have

    positions --> output of either [positionsFromTopLeftCorner] or [positionsFromRaster]

    bg_value -> in order to exclude background when averaging
                set this value to > 0
                if bg_img is given, bg_value should be a threshold between background and signal


    OPTIONAL:
    ---------------
        thresh --> image intensity separating background from signal

        postprocessing_method ...

            'POLY replace' --> replace [arr] with a 2d polynomial fit
            'KW replace'   -->  ...               a fitted Kang-Weiss function
            'AoV replace'  --> ...                a fitted Angle-of-view function

            'POLY repair' --> same as above but either replacing empty
            'KW repair'       areas of smoothing out high gradient
            'AoV repair'      variations (POLY only)

    visualize --> set to True to plot intermediate steps


     max_iter --> maximum number of iterations
     max_dev --> break iteration, if deviation between consecutive
                 vignetting array < [max_dev]

    '''
    assert postprocessing_method in ('POLY replace', 'KW replace',
                                     'AoV replace', 'POLY repair',
                                     'KW repair', 'AoV repair')

    if bg_img is not None:
        bg_img = imread(bg_img)

    (f0, f1), cell_positions, (d0, d1) = positions
    n1, n0 = n_cells_device

    imgs[0] = imread(imgs[0])
    s0, s1 = imgs[0].shape[:2]

    # grid resolution:
    g0, g1 = int(s0 / d0) + 1,  int(s1 / d1) + 1

    # averaged areas in device for all images:
    devices = np.full((len(imgs), n0, n1), np.nan)
    # image to grid for all images:
    grid = np.full((len(imgs), g0, g1), np.nan)

    if thresh is None:
        thresh = bg_value

    # cell average function:
    def fn(x):
        # average excluding background
        ind = x > thresh
        # majority is background:
        if ind.sum() < 0.5 * ind.size:
            return np.nan
        return x[ind].mean()

    gen = _DeviceAndGridSlice(len(imgs), n0, n1, g0, g1, cell_positions)
    # FILL [devices] and [grid]:
    for (img, (n, q0, q1, qq0, qq1)) in zip(imgs, gen.iter()):
        img = imread(img, dtype=float)

        if bg_img is not None:
            img -= bg_img

        p0init, p1init = cell_positions[0]
        p01 = (f0 - d0 * p0init, f1 - d1 * p1init)
        g = subCell2DFnArray(img, fn, (g0, g1),  d01=(d0, d1), p01=p01)
        grid[n] = g
        # assign averaged cell values to device:
        devices[n, q0, q1] = g[qq0, qq1]

        if visualize:
            # for debugging
            _visualize(grid[n], devices[n], img,
                       subCell2DCoords(img, (g0, g1),
                                       p01=p01, d01=(d0, d1)))

    # INITIAL FLAT FIELD:
    ff = np.nanmean(grid, axis=0)
    if bg_img is None:
        ff -= bg_value
    ff /= np.nanmax(ff)
    ffs = np.full_like(grid, np.nan)

    # OBJECT (flat field corrected):
    obj = np.full_like(devices, np.nan)
    n = 0
    # ITERATIVE SAPARATION OR OBJECT AND FLATFIELD:
    while True:
        for (n, q0, q1, qq0, qq1) in gen.iter():
            obj[n, q0, q1] = devices[n, q0, q1] / ff[qq0, qq1]
        avgobj = np.nanmean(obj, axis=0)
        # NEXT FLATFIELD
        for (n, q0, q1, qq0, qq1) in gen.iter():
            ffs[n, qq0, qq1] = devices[n, q0, q1] / avgobj[q0, q1]
        ff2 = np.nanmean(ffs, axis=0)
        # STOP?
        dev = (np.nanmean((ff - ff2)**2))**0.5  # RMS
        ff = ff2
        if n > max_iter or dev < max_dev:
            break
        n += 1
    #####
    # POST PROCESSING included here, because output
    # has different resolution than image
    # TODO: remove post processing
    #####
    # acknowledge that small grid is shifted [p01]:
    yy, xx = offsetMeshgrid(p01, (g0, g1), (s0, s1))
    mask = np.isnan(ff)
    # REPLACE NaN and RESCALE
    if postprocessing_method == 'POLY replace':
        ff2 = polyfit2dGrid(ff, mask, order=5, outgrid=(yy, xx))
    elif postprocessing_method == 'KW replace':
        ff2 = function(ff, mask, outgrid=(yy, xx))
    elif postprocessing_method == 'AoV replace':
        ff2 = function(ff, mask, fn=lambda XY, a:
                       angleOfView(XY, ff.shape, a=a), guess=(0.01),
                       replace_all=True, down_scale_factor=1,
                       outgrid=(yy, xx))
    else:
        if postprocessing_method == 'POLY repair':
            ff = polyfit2dGrid(ff, mask, order=3)
        elif postprocessing_method == 'KW repair':
            ff = function(ff, mask)
        else:  # 'AoV repair'
            ff = function(ff, mask, fn=lambda XY, a:
                          angleOfView(XY, ff.shape, a=a), guess=(0.01),
                          replace_all=False, down_scale_factor=1)
        # rescale:
            # cv2.remap better than scipy_map_coordinated,
            # because of border-interpolation 'reflect':
        ff2 = cv2.remap(ff, xx.astype(np.float32), yy.astype(np.float32),
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_REFLECT)

    pointdensity = len(imgs) - np.sum(np.isnan(grid), axis=0)
    return ff, ff2, avgobj, pointdensity


def _visualize(grid, device, img, gen):
    # for debugging:
    # show intermediate steps of iteration
    # in [vignettingDiscreteSteps]
    import pylab as plt
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('device')
    ax[0].imshow(device, interpolation='none')
    ax[1].set_title('average')
    ax[1].imshow(grid, interpolation='none')
    ax[2].set_title('grid')
    im = ax[2].imshow(img, interpolation='none')
    for x, y in gen:
        ax[2].plot(x, y)
    fig.colorbar(im)
    plt.show()


if __name__ == '__main__':
    # TODO make test case
    pass
