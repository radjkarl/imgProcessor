
import cv2
import numpy as np

import fancytools.math.line as ln


def alignImageAlongLine(img, line, height=15, length=None, fast=False, borderValue=0):
    '''
    return a sub image aligned along given line
    
    @param img -  numpy.2darray input image to get subimage from
    @param line - list of 2 points [x0,y0,x1,y1])
    @param height - height of output array in y
    @param length - width of output array
    @param fast - speed up calculation using nearest neighbour interpolation
    @returns transformed image as numpy.2darray with found line as in the middle
    '''

    height = int(round(height))
    if height % 2 == 0:#->is even number
        height += 1 #only take uneven numbers to have line in middle
    if length is None:
        length = int(round(ln.length(line)) )
    p0,p1 = np.array(line[0:2], dtype=float),np.array(line[2:], dtype=float)
    norm = p1-p0
    norm /= np.linalg.norm(norm)
    norm = -1*norm[::-1]
    p2 = p0 + (p1-p0)*0.5 + norm*height
    
    middleY = (height-1)/2 + 1
    pp0 = [0,middleY]
    pp1 = [length, middleY]
    pp2 = [length*0.5, height]

    pts1 = np.array([p0,p1,p2], dtype=np.float32)
    pts2 = np.array([pp0,pp1,pp2], dtype=np.float32)
    #TRANSFORM:
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img, M, (length, height),
                          flags=cv2.INTER_NEAREST if fast else cv2.INTER_LINEAR,
                          borderValue=borderValue)   
    return dst


if __name__ == '__main__':
    import pylab as plt
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    from imgProcessor.imgIO import imread

    img = imread(PathStr(imgProcessor.__file__).dirname().join(
                'media', 'electroluminescence', 'EL_module_orig.PNG'))
    
    line = (48,325,162,54)#x0,y0,x1,y1
    
    sub = alignImageAlongLine(img, line, height=40)
    
    plt.figure('original')
    plt.imshow(img)
    plt.plot((line[0],line[2]),(line[1],line[03]))
    
    plt.figure('sub image')
    plt.imshow(sub)

    plt.show()