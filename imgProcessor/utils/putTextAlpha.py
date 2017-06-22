'''
Created on Jun 14, 2017

@author: serkgb
'''
import cv2
import numpy as np


def putTextAlpha(img, text, alpha, org, fontFace, fontScale, color,
                 thickness):  # , lineType=None
    '''
    Extends cv2.putText with [alpha] argument
    '''

    x, y = cv2.getTextSize(text, fontFace,
                           fontScale, thickness)[0]

    ox, oy = org

    imgcut = img[oy - y - 3:oy, ox:ox + x]

    if img.ndim == 3:
        txtarr = np.zeros(shape=(y + 3, x, 3), dtype=np.uint8)
    else:
        txtarr = np.zeros(shape=(y + 3, x), dtype=np.uint8)

    cv2.putText(txtarr, text, (0, y), fontFace,
                fontScale, color,
                thickness=thickness
                #, lineType=lineType
                )

    cv2.addWeighted(txtarr, alpha, imgcut, 1, 0, imgcut, -1)
    return img


if __name__ == '__main__':
    import sys
    import pylab as plt

    text = "Funny text inside the box"
    fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    fontScale = 2
    thickness = 3
    img = (np.random.rand(700 * 1000).reshape(700, 1000) * 100).astype(np.uint8)
    org = (20, img.shape[0] - 20)

    for alpha in np.linspace(0, 1, 300):
        img2 = putTextAlpha(img.copy(), text, alpha, org, fontFace, fontScale,
                            (255, 255, 255), thickness
                            #thickness, lineType
                            )

        if 'no_window' in sys.argv:
            break
        print
        cv2.imshow("XXX", img2)
        cv2.waitKey(10)

    if 'no_window' not in sys.argv:
        cv2.waitKey(0)
