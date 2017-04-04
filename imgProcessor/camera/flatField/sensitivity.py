from imgProcessor.utils.getBackground import getBackground
from imgProcessor.imgIO import imread
from imgProcessor.filters.fastMean import fastMean
from scipy.ndimage.filters import median_filter


def sensitivity(imgs, bg=None):
    '''
    Extract pixel sensitivity from a set of homogeneously illuminated images

    This method is detailed in Section 5 of:
    ---
    K.Bedrich, M.Bokalic et al.:
    ELECTROLUMINESCENCE IMAGING OF PV DEVICES:
    ADVANCED FLAT FIELD CALIBRATION,2017
    ---

    '''
    bg = getBackground(bg)
    for n, i in enumerate(imgs):
        i = imread(i, dtype=float)
        i -= bg
        smooth = fastMean(median_filter(i, 3))
        i /= smooth
        if n == 0:
            out = i
        else:
            out += i
    out /= (n + 1)
    return out