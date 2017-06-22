import numpy as np


def boolMasksToImage(masks):
    '''
    Transform at maximum 8 bool layers --> 2d arrays, dtype=(bool,int)
    to one 8bit image
    '''
    assert len(masks) <= 8, 'can only transform up to 8 masks into image'
    masks = np.asarray(masks, dtype=np.uint8)
    assert masks.ndim == 3, 'layers need to be stack of 2d arrays'
    return np.packbits(masks, axis=0)[0].T


def imageToBoolMasks(arr):
    '''inverse of [boolMasksToImage]'''
    assert arr.dtype == np.uint8, 'image needs to be dtype=uint8'
    masks = np.unpackbits(arr).reshape(*arr.shape, 8)
    return np.swapaxes(masks, 2, 0)


if __name__ == '__main__':
    import pylab as plt
    import sys

    # generate some bool arrays:
    m0 = np.random.randint(0, 2, size=100).reshape(10, 10)
    m1 = np.random.randint(0, 2, size=100).reshape(10, 10)
    m2 = np.random.randint(0, 2, size=100).reshape(10, 10)

    # transform those array to an 8bit image:
    img = boolMasksToImage((m0, m1, m2))

    # transform image back to bool arrays:
    mm0, mm1, mm2 = imageToBoolMasks(img)[:3]

    # prove, that forth and back transformation does not modify data:
    assert np.alltrue(m0 == mm0) and np.alltrue(
        m1 == mm1) and np.alltrue(m2 == mm2)

    if 'no_window' not in sys.argv:
        f, axarr = plt.subplots(3)
        f.suptitle('input bin arrays')
        axarr[0].imshow(m0)
        axarr[1].imshow(m1)
        axarr[2].imshow(m2)

        plt.figure('generated 8bit image')
        plt.imshow(img)

        f, axarr = plt.subplots(3)
        f.suptitle('transformed back')
        axarr[0].imshow(mm0)
        axarr[1].imshow(mm1)
        axarr[2].imshow(mm2)

        plt.show()
