import numpy as np
import cv2
from imgProcessor.utils.sortCorners import sortCorners


def simplePerspectiveTransform(img, quad, shape=None,
                               interpolation=cv2.INTER_LINEAR,
                               inverse=False):
    p = sortCorners(quad).astype(np.float32)
    if shape is not None:
        height, width = shape
    else:
        # get output image size from avg. quad edge length
        width = int(round(0.5 * (np.linalg.norm(p[0] - p[1]) +
                                 np.linalg.norm(p[3] - p[2]))))
        height = int(round(0.5 * (np.linalg.norm(p[1] - p[2]) +
                                  np.linalg.norm(p[0] - p[3]))))

    dst = np.float32([[0,     0],
                      [width, 0],
                      [width, height],
                      [0,     height]])

    if inverse:
        s0, s1 = img.shape[:2]
        dst /= ((width / s1), (height / s0))
        H = cv2.getPerspectiveTransform(dst, p)
    else:
        H = cv2.getPerspectiveTransform(p, dst)

    return cv2.warpPerspective(img, H, (width, height), flags=interpolation)


if __name__ == '__main__':
    import imgProcessor
    from imgProcessor.imgIO import imread
    from fancytools.os.PathStr import PathStr
    import pylab as plt
    import sys

    # corner points (x,y):
    points = ((61, 15),
              (748, 19),
              (747, 697),
              (57, 703))
    # image:
    path = PathStr(imgProcessor.__file__).dirname().join(
        'media', 'electroluminescence', 'EL_cell_cracked.png')

    #######
    img = imread(path)
    img2 = simplePerspectiveTransform(img, points)
    img3 = simplePerspectiveTransform(img2, points,
                                      inverse=True, shape=img.shape)
    #######
    if 'no_window' not in sys.argv:
        p = np.array(points)

        plt.figure('input')
        plt.imshow(img)
        plt.scatter(p[:, 0], p[:, 1], color='r')
        plt.figure('output')
        plt.imshow(img2)
        plt.figure('inverse')
        plt.imshow(img3)
        plt.show()
