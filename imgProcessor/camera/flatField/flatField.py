import numpy as np

from scipy.ndimage.filters import gaussian_filter, median_filter

from skimage.transform import resize


from imgProcessor.camera.flatField.flatFieldFromCalibration import flatFieldFromCalibration
from imgProcessor.camera.flatField.vignettingFromMultipleSpots import vignettingFromMultipleSpots
from imgProcessor.camera.flatField.vignettingFromDifferentObjects import vignettingFromDifferentObjects
from imgProcessor.camera.flatField.ObjectVignettingSeparation import vignettingFromSameObject

from imgProcessor.camera.flatField import interpolationMethods


VIGNETTING_MODELS = {'multiple_spots': vignettingFromMultipleSpots,
                     'different_objects': vignettingFromDifferentObjects,
                     'same_object': vignettingFromSameObject}

INTERPOLATION_METHODS = {'kangWeiss': interpolationMethods.function,
                         'polynomial': interpolationMethods.polynomial}


def flatField(closeDist_img=None, inPlane_img=None,
              closeDist_bg=None, inPlane_bg=None,
              vignetting_model='different_objects',
              interpolation_method='kangWeiss',
              inPlane_scale_factor=None):

    # 1. Pixel sensitivity:
    if closeDist_img is not None:
        # TODO: find better name
        ff1 = flatFieldFromCalibration(closeDist_img, closeDist_bg)
    else:
        ff1 = 0

    # 2. Vignetting from in-plane measurements:
    if inPlane_img is not None:
        bg = gaussian_filter(median_filter(ff1, 3), 9)
        ff1 -= bg

        ff2, mask = VIGNETTING_MODELS[vignetting_model](inPlane_img,
                                                        inPlane_bg, inPlane_scale_factor)
#         import pylab as plt
#         plt.imshow(mask)
#         plt.show()
        ff2smooth = INTERPOLATION_METHODS[interpolation_method](ff2, mask)
        if isinstance(ff1, np.ndarray) and ff1.shape != ff2smooth.shape:
            ff2smooth = resize(ff2smooth, ff1.shape, mode='reflect')
    else:
        ff2 = 0
        ff2smooth = 0

    return ff1 + ff2smooth, ff2
