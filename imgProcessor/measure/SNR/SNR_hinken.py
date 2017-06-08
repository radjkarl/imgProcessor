from __future__ import division

import numpy as np
from imgProcessor.imgIO import imread



def SNR_hinken(imgs, bg=0, roi=None):
    '''
    signal-to-noise ratio (SNR) as mean(images) / std(images)
    as defined in Hinken et.al. 2011 (DOI: 10.1063/1.3541766)
    
    works on unloaded images
    no memory overload if too many images are given
    '''
    mean = None
    M = len(imgs)
    if bg is not 0:
        bg = imread(bg)[roi]
        if roi is not None:
            bg = bg[roi]
    #calc mean:
    for i in imgs:
        img = imread(i).asfarray()
        if roi is not None:
            img = img[roi]
        img -= bg
        if mean is None:
            #init
            mean = np.zeros_like(img)
            std = np.zeros_like(img)
        mean += img
        del img
    mean /= M
    #calc std of mean:
    for i in imgs:
        img = imread(i).asfarray()
        if roi is not None:
            img = img[roi]
        img -= bg
        std += (mean - img)**2
        del img
    std = (std / M)**0.5
    return mean.mean() / std.mean()