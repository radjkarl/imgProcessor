from scipy.ndimage import median_filter
import numpy as np



def medianThreshold(img, threshold=0.1, size=3, condition='>', copy=True):
    '''
    set every the pixel value of the given [img] to the median filtered one
    of a given kernel [size]
    in case the relative [threshold] is exeeded 
    condition = '>' OR '<'
    '''
    indices = None
    if threshold > 0:
        blur = median_filter(img, size=size)
      
        with np.errstate(divide='ignore',invalid='ignore', over='ignore'):
            
            if condition == '>':
                indices = abs((img-blur)/blur) > threshold
            else:
                indices = abs((img-blur)/blur) < threshold
    
        if copy:
            img = img.copy()
    
        img[indices] = blur[indices]
    return img, indices



if __name__ == '__main__':
    import sys
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    from imgProcessor.imgIO import imread

    img = imread(PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence', 'EL_module_orig.PNG'))
    
    med,ind = medianThreshold(img)
    
    if 'no_window' not in sys.argv:
        plt.figure('original')
        plt.imshow(img)
        plt.figure('filtered')
        plt.imshow(med)
        plt.figure('filtered indices')
        plt.imshow(ind)
        plt.show()