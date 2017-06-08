# coding=utf-8
from imgProcessor.camera.DarkCurrentMap import averageSameExpTimes
from imgProcessor.imgIO import imread

import numpy as np


def getBackground(bgImages, **kwargs):
    # AVERAGE BACKGROUND IMAGES IF MULTIPLE ARE GIVEN:
    if bgImages is not None:
        if (type(bgImages) in (tuple, list)
                or isinstance(bgImages, np.ndarray) and bgImages.ndim == 3):
            if len(bgImages) > 1:
                return averageSameExpTimes(bgImages)
            else:
                return imread(bgImages[0], **kwargs)
        else:
            return imread(bgImages, **kwargs)
    else:
        return 0
