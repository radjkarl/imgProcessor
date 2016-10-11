from __future__ import print_function

import numpy as np

from skimage.measure import label
from skimage.filters import threshold_otsu

from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.measurements import center_of_mass

# from fancytools.fit.fit2dArrayToFn import fit2dArrayToFn
#
# from imgProcessor.equations.vignetting import vignetting, guessVignettingParam
from imgProcessor.imgIO import imread
from imgProcessor.utils.getBackground import getBackground


def vignettingFromMultipleSpots(
        images, bgImages=None, averageSpot=True, thresh=None):
    '''
    Through fitting in image plane images spots
    (e.g. created with a mobile phone at different positions)
    to a vignetting function
    Args:
        averageSpot(bool): True: take only the average intensity of each spot
        thresh(float): marks the minimum spot value (estimated with Otsus method otherwise)
    Returns:
        flat-field correction map
    '''

    avgBg = getBackground(bgImages)

    fitimg, mask = None, None
    mx = 0
    for c, img in enumerate(images):
        print('%s/%s' % (c + 1, len(images)))

        img = imread(img, dtype=float) - avgBg
        # init:
        if fitimg is None:
            fitimg = np.zeros_like(img)
            mask = np.zeros_like(img, dtype=bool)
        # find spot:
        if thresh is None:
            t = threshold_otsu(img)
        else:
            t = thresh
        # take brightest spot
        spots, n = label(minimum_filter(img > t, 3),
                         background=0, return_num=True)
        spot_sizes = [(spots == i).sum() for i in range(n)]
        spot = spots == np.argmax(spot_sizes)

        if averageSpot:
            spot = center_of_mass(spot)
            mx2 = fitimg[spot] = img[spot].mean()
        else:
            fitimg[spot] = img[spot]
            mx2 = img[spot].max()

        if mx2 > mx:
            mx = mx2

        mask[spot] = 1

    # scale fitimg:
    fitimg /= mx

    return fitimg, ~mask

#
#     flatfield = fit2dArrayToFn(fitimg, vignetting, mask=mask,
#                                down_scale_factor=1 if averageSpot else 0.3,
#                                guess=guessVignettingParam(fitimg))[0]
#
#     return flatfield, fitimg


# TODO: make generic example

#
#     if 'no_window' not in sys.argv:
#         plt.figure('example vignetting img (1/10)')
#         plt.imshow(vigs[0])
#         plt.colorbar()
#
#         plt.figure('example vignetting img (10/10)')
#         plt.imshow(vigs[-1])
#         plt.colorbar()
#
#         plt.figure('averaged vignetting array')
#         plt.imshow(avg)
#         plt.colorbar()
#
#         plt.figure('standard deviation')
#         plt.imshow(std)
#         plt.colorbar()
#
#         plt.show()
