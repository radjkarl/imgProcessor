
from imgProcessor.utils.getBackground import getBackground
from imgProcessor.imgIO import imread
from imgProcessor.filters.fastMean import fastMean
from scipy.ndimage.filters import median_filter



def sensitivity(imgs, bg=None):
    bg = getBackground(bg)
    for n, i in enumerate(imgs):
        i = imread(i, dtype=float)
        i -=bg
        smooth =  fastMean(median_filter(i, 3))
        i/=smooth
        if n == 0:
            out = i
        else:
            out+= i
    out /= (n+1)
    return out
        
        

        
        
    