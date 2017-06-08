from imgProcessor.imgSignal import scaleSignalCutParams
from imgProcessor.transform.imgAverage import imgAverage


def getBackground2(bgImages=None, img=None):
    if bgImages is None:
        return scaleSignalCutParams(img, 0.01)[0]
    elif type(bgImages) in (int, float):
        return bgImages
    else:
        return imgAverage(bgImages)
    return 0.0