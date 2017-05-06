import numpy as np
from imgProcessor.transform.simplePerspectiveTransform \
    import simplePerspectiveTransform


def rmBorder(img, border=None):
    '''
    border               [None], if images are corrected and device ends at
                               image border
                         [one number] (like 50), 
                               if there is an equally spaced border
                               around the device
                         [two tuples] like ((50,60),(1500,900))
                             means ((Xfrom,Yfrom),(Xto, Yto))
                         [four tuples] like ((x0,y0),(x1,y1),...(x3,y3))
    '''
    if border is None:
        pass
    elif len(border) == 2:
        s0 = slice(border[0][1], border[1][1])
        s1 = slice(border[0][0], border[1][0])
        img = img[s0, s1]
    elif len(border) == 4:
        # eval whether border values are orthogonal:
        x = np.unique(border[:, 0])
        y = np.unique(border[:, 1])
        if len(x) == 2 and len(y) == 2:
            s0 = slice(y[0], y[1])
            s1 = slice(x[0], x[1])
            img = img[s0, s1]
        else:
            # edges are irregular:
            img = simplePerspectiveTransform(img, border)
    else:
        raise Exception('[border] input wrong')
    return img
