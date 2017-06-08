from PyQt5 import QtSvg, QtGui, QtCore
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2
import numpy as np

from imgProcessor.camera.LensDistortion import LensDistortion
from fancytools.math.rotatePolygon import rotatePolygon

import imgProcessor
from imgProcessor.reader.qImageToArray import qImageToArray

from fancytools.os.PathStr import PathStr
from imgProcessor.exceptions import NothingFound
MEDIA_PATH = PathStr(imgProcessor.__file__).dirname().join('media')

# =============================================================================
# an old comparison of systematic measurement errors of different lens
# calibration methods
# not further developed
# =============================================================================


def simulateSytematicError(N_SAMPLES=5, N_IMAGES=10,
                           SHOW_DETECTED_PATTERN=True,  # GRAYSCALE=False,
                           HEIGHT=500, PLOT_RESULTS=True, PLOT_ERROR_ARRAY=True,
                           CAMERA_PARAM=None, PERSPECTIVE=True, ROTATION=True,
                           RELATIVE_PATTERN_SIZE=0.5, POSITION=True,
                           NOISE=25, BLUR=(3, 3), PATTERNS=None):
    '''
    Simulates a lens calibration using synthetic images
    * images are rendered under the given HEIGHT resolution
    * noise and smoothing is applied
    * perspective and position errors are applied
    * images are deformed using the given CAMERA_PARAM

    * the detected camera parameters are used to calculate the error to the given ones


    simulation
    -----------
    N_IMAGES -> number of images to take for a camera calibration
    N_SAMPLES -> number of camera calibrations of each pattern type

    output
    --------
    SHOW_DETECTED_PATTERN: print each image and detected pattern to screen
    PLOT_RESULTS: plot boxplots of the mean error and std of the camera parameters
    PLOT_ERROR_ARRAY: plot position error for the lens correction

    pattern
    --------
    this simulation tests the openCV standard patterns: chess board, asymmetric and symmetric circles

    GRAYSCALE: whether to load the pattern as gray scale
    RELATIVE_PATTERN_SIZE: the relative size of the pattern within the image (0.4->40%)
    PERSPECTIVE: [True] -> enable perspective distortion
    ROTATION: [True] -> enable rotation of the pattern
    BLUR: False or (sizex,sizey), like (3,3)

    CAMERA_PARAM: camera calibration parameters as [fx,fy,cx,cy,k1,k2,k3,p1,p2]


    '''
    print(
        'calculate systematic error of the implemented calibration algorithms')

    # LOCATION OF PATTERN IMAGES
    folder = MEDIA_PATH

    if PATTERNS is None:
        PATTERNS = ('Chessboard', 'Asymmetric circles', 'Symmetric circles')

    patterns = OrderedDict((  # n of inner corners
        ('Chessboard',        ((6, 9), 'chessboard_pattern_a3.svg')),
        ('Asymmetric circles', ((4, 11), 'acircles_pattern_a3.svg')),
        ('Symmetric circles', ((8, 11), 'circles_pattern_a3.svg')),
    ))
    # REMOVE PATTERNS THAT ARE NOT TO BE TESTED:
    [patterns.pop(key) for key in patterns if key not in PATTERNS]

    if SHOW_DETECTED_PATTERN:
        cv2.namedWindow('Pattern', cv2.WINDOW_NORMAL)
    # number of positive detected patterns:
    success = []
    # list[N_SAMPLES] of random camera parameters
    fx, fy, cx, cy, k1, k2, k3, p1, p2 = [], [], [], [], [], [], [], [], []
    # list[Method, N_SAMPLES] of given-detected parameters:
    errl, fxl, fyl, cxl, cyl, k1l, k2l, k3l, p1l, p2l = [
    ], [], [], [], [], [], [], [], [], []
    # list[Method, N_SAMPLES] of magnitude(difference of displacement vector
    # array):
    dxl = []
    dyl = []
    # maintain aspect ratio of din a4, a3...:
    aspect_ratio_DIN = 2.0**0.5
    width = int(round(HEIGHT / aspect_ratio_DIN))

    if CAMERA_PARAM is None:
        CAMERA_PARAM = [
            HEIGHT, HEIGHT, HEIGHT / 2, width / 2, 0.0, 0.01, 0.1, 0.01, 0.001]

    # ???CREATE N DIFFERENT RANDOM LENS ERRORS:
    for n in range(N_SAMPLES):
        # TODO: RANDOMIZE CAMERA ERROR??
        fx.append(CAMERA_PARAM[0])  # * np.random.uniform(1, 2) )
        fy.append(CAMERA_PARAM[1])  # * np.random.uniform(1, 2) )
        cx.append(CAMERA_PARAM[2])  # * np.random.uniform(0.9, 1.1) )
        cy.append(CAMERA_PARAM[3])  # * np.random.uniform(0.9, 1.1) )
        k1.append(CAMERA_PARAM[4])  # + np.random.uniform(-1, 1)*0.1)
        k2.append(CAMERA_PARAM[5])  # + np.random.uniform(-1, 1)*0.01)
        p1.append(CAMERA_PARAM[6])  # + np.random.uniform(0, 1)*0.1)
        p2.append(CAMERA_PARAM[7])  # + np.random.uniform(0, 1)*0.01)
        k3.append(CAMERA_PARAM[8])  # + np.random.uniform(0, 1)*0.001)

    L = LensDistortion()
    # FOR EVERY METHOD:
    for method, (board_size, filename) in patterns.items():

        f = folder.join(filename)

        # LOAD THE SVG FILE, AND SAVE IT WITH NEW RESOLUTION:
        svg = QtSvg.QSvgRenderer(f)
        image = QtGui.QImage(width * 4, HEIGHT * 4, QtGui.QImage.Format_ARGB32)
        image.fill(QtCore.Qt.white)
        # Get QPainter that paints to the image
        painter = QtGui.QPainter(image)
        svg.render(painter)
        # Save, image format based on file extension
#         f = "rendered.png"
#         image.save(f)
#
#         if GRAYSCALE:
#             img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#         else:
#             img = cv2.imread(f)

        img = qImageToArray(image)

        success.append([])
        fxl.append([])
        errl.append([])
        fyl.append([])
        cxl.append([])
        cyl.append([])
        k1l.append([])
        k2l.append([])
        k3l.append([])
        p1l.append([])
        p2l.append([])

        dxl.append([])
        dyl.append([])

        imgHeight, imgWidth = img.shape[0], img.shape[1]

        for n in range(N_SAMPLES):
            L.calibrate(board_size, method)

            print('SET PARAMS:', fx[n], fy[n], cx[n],
                  cy[n], k1[n], k2[n], k3[n], p1[n], p2[n])
            L.setCameraParams(
                fx[n], fy[n], cx[n], cy[n], k1[n], k2[n], k3[n], p1[n], p2[n])
            L._coeffs['shape'] = (imgHeight, imgWidth)

            hw = imgWidth * 0.5
            hh = imgHeight * 0.5
            for m in range(N_IMAGES):
                pts1 = np.float32([[hw, hh + 100],
                                   [hw - 100, hh - 100],
                                   [hw + 100, hh - 100]])
                pts2 = pts1.copy()
                if ROTATION:
                    rotatePolygon(pts2, np.random.randint(0, 2 * np.pi))

                if PERSPECTIVE:
                    # CREATE A RANDOM PERSPECTIVE:
                    pts2 += np.random.randint(-hw *
                                              0.05, hh * 0.05, size=(3, 2))
                    # MAKE SURE THAT THE PATTERN IS FULLY WITHIN THE IMAGE:
                pts2 *= RELATIVE_PATTERN_SIZE

                # MOVE TO THE CENTER
                pts2[:, 0] += hw * (1 - RELATIVE_PATTERN_SIZE)
                pts2[:, 1] += hh * (1 - RELATIVE_PATTERN_SIZE)

                if POSITION:
                    f = ((2 * np.random.rand(2)) - 1)
                    pts2[:, 0] += hw * 0.7 * f[0] * (1 - RELATIVE_PATTERN_SIZE)
                    pts2[:, 1] += hh * 0.7 * f[1] * (1 - RELATIVE_PATTERN_SIZE)
                # EXEC PERSPECTICE, POSITION, ROTATION:
                M = cv2.getAffineTransform(pts1, pts2)
                img_warped = cv2.warpAffine(
                    img, M, (imgWidth, imgHeight), borderValue=(230, 230, 230))
                # DOWNSCALE IMAGE AGAIN - UPSCALING AND DOWNSCALING SHOULD BRING THE ERRROR
                # WARPING DOWN
                img_warped = cv2.resize(img_warped, (width, HEIGHT))
                # CREATE THE LENS DISTORTION:
                mapx, mapy = L.getDistortRectifyMap(width, HEIGHT)
                # print 664, mapx.shape
                img_distorted = cv2.remap(
                    img_warped, mapx, mapy, cv2.INTER_LINEAR, borderValue=(230, 230, 230))

#                 img_distorted[img_distorted==0]=20
#                 img_distorted[img_distorted>100]=230
                if BLUR:
                    img_distorted = cv2.blur(img_distorted, BLUR)
                if NOISE:
                    # soften, black and white more gray, and add noise
                    img_distorted = img_distorted.astype(np.int16)
                    img_distorted += (np.random.rand(*img_distorted.shape) *
                                      NOISE).astype(img_distorted.dtype)
                    img_distorted = np.clip(
                        img_distorted, 0, 255).astype(np.uint8)
#                 plt.imshow(img_distorted)
#                 plt.show()
                found = L.addImg(img_distorted)

                if SHOW_DETECTED_PATTERN and found:
                    img_distorted = L.drawChessboard(img_distorted)
                    cv2.imshow('Pattern', img_distorted)
                    cv2.waitKey(1)

            success[-1].append(L.findCount)
            try:
                L._coeffs = None
                errl[-1].append(L.coeffs['reprojectionError'])

                L.correct(img_distorted)

                c = L.getCameraParams()
                print('GET PARAMS:', c)

                fxl[-1].append(fx[n] - c[0])
                fyl[-1].append(fy[n] - c[1])
                cxl[-1].append(cx[n] - c[2])
                cyl[-1].append(cy[n] - c[3])
                k1l[-1].append(k1[n] - c[4])
                k2l[-1].append(k2[n] - c[5])
                k3l[-1].append(k3[n] - c[6])
                p1l[-1].append(p1[n] - c[7])
                p2l[-1].append(p2[n] - c[8])

                if PLOT_ERROR_ARRAY:
                    dx = (mapx - L.mapx) / 2
                    dy = (mapy - L.mapy) / 2
                    dxl[-1].append(dx)
                    dyl[-1].append(dy)

            except NothingFound:
                print(
                    "Couldn't create a calibration because no patterns were detected")

        del painter

    # AVERAGE SAMPLES AND GET STD
    dx_std, dx_mean = [], []
    dy_std, dy_mean = [], []
    mag = []
    std = []
    for patterndx, patterndy in zip(dxl, dyl):
        x = np.mean(patterndx, axis=0)
        dx_mean.append(x)
        y = np.mean(patterndy, axis=0)
        dy_mean.append(y)
        x = np.std(patterndx, axis=0)
        mag.append((x**2 + y**2)**0.5)
        dx_std.append(x)
        y = np.std(patterndy, axis=0)
        dy_std.append(y)
        std.append((x**2 + y**2)**0.5)

    # PLOT
    p = len(patterns)
    if PLOT_RESULTS:
        fig, axs = plt.subplots(nrows=2, ncols=5)

        axs = np.array(axs).ravel()
        for ax, typ, tname in zip(axs,
                                  (success, fxl, fyl, cxl, cyl,
                                   k1l, k2l, k3l, p1l, p2l),
                                  ('Success rate', 'fx', 'fy', 'cx',
                                   'cy', 'k1', 'k2', 'k3', 'p1', 'p2')
                                  ):
            ax.set_title(tname)
            # , showmeans=True, meanline=True)#labels=patterns.keys())
            ax.boxplot(typ, notch=0, sym='+', vert=1, whis=1.5)
            # , ha=ha[n])
            ax.set_xticklabels(patterns.keys(), rotation=40, fontsize=8)

    if PLOT_ERROR_ARRAY:

        mmin = np.min(mag)
        mmax = np.max(mag)
        smin = np.min(std)
        smax = np.max(std)

        plt.figure()
        for n, pattern in enumerate(patterns.keys()):

            plt.subplot(int('2%s%s' % (p, n + 1)), axisbg='g')
            plt.title(pattern)
            plt.imshow(mag[n], origin='upper', vmin=mmin, vmax=mmax)
            if n == p - 1:
                plt.colorbar(label='Average')

            plt.subplot(int('2%s%s' % (p, n + p + 1)), axisbg='g')
            plt.title(pattern)
            plt.imshow(std[n], origin='upper', vmin=smin, vmax=smax)
            if n == p - 1:
                plt.colorbar(label='Standard deviation')

        fig = plt.figure()
        fig.suptitle('Individually scaled')
        for n, pattern in enumerate(patterns.keys()):
            # downscale - show max 30 arrows each dimension
            sy, sx = dx_mean[n].shape
            ix = int(sx / 15)
            if ix < 1:
                ix = 1
            iy = int(sy / 15)
            if iy < 1:
                iy = 1

            Y, X = np.meshgrid(np.arange(0, sy, iy), np.arange(0, sx, ix))

            plt.subplot(int('2%s%s' % (p, n + 1)), axisbg='g')
            plt.title(pattern)
            plt.imshow(mag[n], origin='upper')
            plt.colorbar()
            plt.quiver(
                X, Y, dy_mean[n][::ix, ::iy] * 20, dx_mean[n][::ix, ::iy] * 20)

            plt.subplot(int('2%s%s' % (p, n + p + 1)), axisbg='g')
            plt.title(pattern)
            plt.imshow(std[n], origin='upper')
            plt.colorbar()
            # plt.quiver(X,Y,dx_std[n][::ix,::iy]*50, dy_std[n][::ix,::iy]*10)

        #############################################
        fig = plt.figure()
        fig.suptitle('Spatial uncertainty + deflection')

        for n, pattern in enumerate(patterns.keys()):
            L.calibrate(board_size, method)
            # there is alot of additional calc thats not necassary:
            L.setCameraParams(
                fx[0], fy[0], cx[0], cy[0], k1[0], k2[0], k3[0], p1[0], p2[0])
            L._coeffs['shape'] = (imgHeight, imgWidth)
            L._coeffs['reprojectionError'] = np.mean(errl[n])


#             deflection_x, deflection_y = L.getDeflection(width, HEIGHT)
#             deflection_x += dx_mean[n]
#             deflection_y += dy_mean[n]

            ux, uy = L.standardUncertainties()

            plt.subplot(int('2%s%s' % (p, n + 1)), axisbg='g')
            plt.title(pattern)
            plt.imshow(mag[n], origin='upper')
            plt.colorbar()

            # DEFLECTION
            plt.subplot(int('2%s%s' % (p, n + p + 1)), axisbg='g')
            plt.title(pattern)
            plt.imshow(np.linalg.norm([ux, uy], axis=0), origin='upper')
            plt.colorbar()
            # DEFL: VECTORS
            # downscale - show max 30 arrows each dimension
            sy, sx = dx_mean[n].shape
            ix = int(sx / 15)
            if ix < 1:
                ix = 1
            iy = int(sy / 15)
            if iy < 1:
                iy = 1
            Y, X = np.meshgrid(np.arange(0, sy, iy), np.arange(0, sx, ix))
            plt.quiver(X, Y, ux[::ix, ::iy] * 20, uy[::ix, ::iy] * 20)

    if PLOT_ERROR_ARRAY or PLOT_RESULTS:
        plt.show()

    return dx_mean, dy_mean


if __name__ == '__main__':
    import sys
    if 'no_window' not in sys.argv:
        simulateSytematicError()
