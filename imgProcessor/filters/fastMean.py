
import cv2


def fastMean(img, f=10, inplace=False):
    '''
    for bigger ksizes it if often faster to resize an image
    rather than blur it...
    '''
    s0,s1 = img.shape[:2]
    ss0 = int(round(s0/f))
    ss1 = int(round(s1/f))

    small = cv2.resize(img,(ss1,ss0), interpolation=cv2.INTER_AREA)
        #bigger
    k = {'interpolation':cv2.INTER_LINEAR}
    if inplace:
        k['dst']=img
    return cv2.resize(small,(s1,s0), **k)

if __name__ == '__main__':
    import numpy as np
    import pylab as plt
    import sys
    
    shape = (1000, 700)
    center = (350, 500)
    radius = 200
    noise = 10
    
    img = np.zeros(shape, dtype=np.uint8)
    cv2.circle(img, center, radius, color=100, thickness=-1)
    img+= (np.random.rand(shape[0]*shape[1])*noise).reshape(shape
                                            ).astype(img.dtype)
    
    
    img2 = fastMean(img, 20)

    if 'no_window' not in sys.argv:
        plt.figure('original')
        plt.imshow(img)
        plt.figure('smoothed')
        plt.imshow(img2)
        plt.show()