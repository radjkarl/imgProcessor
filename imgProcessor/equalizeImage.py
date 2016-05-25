import numpy as np
import cv2
from skimage import exposure

from fancytools.os.PathStr import PathStr


#TODO: check whether name_additive and save_path still needed
def equalizeImage(img, save_path=None, name_additive='_eqHist'):
    '''
    Equalize the histogram (contrast) of an image
    works with RGB/multi-channel images
    and flat-arrays
    
    @param img  - image_path or np.array
    @param save_path if given output images will be saved there
    @param name_additive if given this additive will be appended to output images 

    @return output images if input images are numpy.arrays and no save_path is given
    @return None elsewise 
    '''

    if isinstance(img, basestring):
        img = PathStr(img)
        if not img.exists():
            raise Exception("image path doesn't exist")
        img_name = img.basename().replace('.','%s.' %name_additive)
        if save_path is None:
            save_path = img.dirname()
        img = cv2.imread(img)

    
    if img.dtype != np.dtype('uint8'):
        #openCV cannot work with float arrays or uint > 8bit
        eqFn = _equalizeHistogram
    else:
        eqFn = cv2.equalizeHist
    if len(img.shape) == 3:#multi channel img like rgb
        for i in range(img.shape[2]):
            img[:, :, i] = eqFn(img[:, :, i]) 
    else: # grey scale image 
        img = eqFn(img)
    if save_path:
        img_name = PathStr(save_path).join(img_name)
        cv2.imwrite(img_name, img)
    return img


def _equalizeHistogram(img):
    '''
    histogram equalisation not bounded to int() or an image depth of 8 bit
    works also with negative numbers
    '''
    #to float if int:
    intType = None
    if 'f' not in img.dtype.str:
        TO_FLOAT_TYPES = {  np.dtype('uint8'):np.float16,
                    np.dtype('uint16'):np.float32,
                    np.dtype('uint32'):np.float64,
                    np.dtype('uint64'):np.float64} 

        intType = img.dtype
        img = img.astype(TO_FLOAT_TYPES[intType], copy=False)
    
    #get image deph
    DEPTH_TO_NBINS = {np.dtype('float16'):256, #uint8
                      np.dtype('float32'):32768, #uint16
                      np.dtype('float64'):2147483648} #uint32
    nBins = DEPTH_TO_NBINS[img.dtype]

    #scale to -1 to 1 due to skikit-image restrictions
    mn, mx = np.amin(img), np.amax(img)
    if abs(mn) > abs(mx):
        mx = mn
    img /= mx
    img = exposure.equalize_hist(img, nbins=nBins)
    
    img *= mx
    
    if intType:
        img = img.astype(intType)
    return img



if __name__ == '__main__':
    import pylab as plt
    import imgProcessor
    from imgProcessor.imgIO import imread

    img = imread(PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence', 'EL_module_orig.PNG'))
    
    eq = equalizeImage(img.copy())
    plt.figure('original')
    plt.imshow(img)

    plt.figure('equalised histogram')
    plt.imshow(eq)

    plt.show()