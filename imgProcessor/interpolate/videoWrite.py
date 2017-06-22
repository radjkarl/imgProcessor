'''
Created on Jun 14, 2017

@author: serkgb
'''
import cv2
import numpy as np
from imgProcessor.utils.putTextAlpha import putTextAlpha
from imgProcessor.interpolate.InterpolateImageStack \
    import LinearInterpolateImageStack
from pyqtgraph.functions import makeRGBA


def videoWrite(path, imgs, levels=None, shape=None, frames=15,
               annotate_names=None,
               lut=None, updateFn=None):
    '''
    TODO
    '''
    frames = int(frames)
    if annotate_names is not None:
        assert len(annotate_names) == len(imgs)

    if levels is None:
        if imgs[0].dtype == np.uint8:
            levels = 0, 255
        elif imgs[0].dtype == np.uint16:
            levels = 0, 2**16 - 1
        else:
            levels = np.min(imgs), np.max(imgs)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = imgs.shape[1:3]

    if shape and shape != (h, w):
        h, w = shape
        imgs = [cv2.resize(i, (w, h)) for i in imgs]

    assert path[-3:] in ('avi',
                         'png'), 'video export only supports *.avi or *.png'
    isVideo = path[-3:] == 'avi'
    if isVideo:
        cap = cv2.VideoCapture(0)
        # im.ndim==4)
        out = cv2.VideoWriter(path, fourcc, frames, (w, h), isColor=1)

    times = np.linspace(0, len(imgs) - 1, len(imgs) * frames)
    interpolator = LinearInterpolateImageStack(imgs)

    if lut is not None:
        lut = lut(imgs[0])

    for n, time in enumerate(times):
        if updateFn:
            # update progress:
            updateFn.emit(100 * n / len(times))
        image = interpolator(time)

        cimg = makeRGBA(image, lut=lut,
                        levels=levels)[0]
        cimg = cv2.cvtColor(cimg, cv2.COLOR_RGBA2BGR)

        if annotate_names:
            text = annotate_names[n // frames]
            alpha = 0.5
            org = (0, cimg.shape[0])
            fontFace = cv2.FONT_HERSHEY_PLAIN
            fontScale = 2
            thickness = 3
            putTextAlpha(cimg, text, alpha, org, fontFace, fontScale,
                         (0, 255, 0), thickness
                         )

        if isVideo:
            out.write(cimg)
        else:
            cv2.imwrite('%s_%i_%.3f.png' % (path[:-4], n, time), cimg)

    if isVideo:
        cap.release()
        out.release()


if __name__ == '__main__':
    # TODO
    pass
