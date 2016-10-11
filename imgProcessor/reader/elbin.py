from __future__ import print_function

import numpy as np


def elbin(filename):
    '''
    Read EL images (*.elbin) created by the RELTRON EL Software
    http://www.reltron.com/Products/Solar.html
    '''
#     arrs = []
    labels = []

    # These are all exposure times [s] to be selectable:
    TIMES = (0.3, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2, 4.8, 6.4, 9.6, 12.8, 19.2,
             25.6, 38.4, 51.2, 76.8, 102.6, 153.6, 204.6, 307.2, 409.8, 614.4,
             819., 1228.8, 1638.6, 3276.6, 5400., 8100., 12168., 18216., 27324.,
             41004., 61488., 92268.)

    with open(filename, 'rb') as f:
        # image shape and number:
        height, width, frames = np.frombuffer(f.read(4 * 3), dtype=np.uint32)
        arrs = np.empty((frames, width, height), dtype=np.uint16)
        for i in range(frames):
            # read header between all frames:
            current, voltage = np.frombuffer(f.read(8 * 2), dtype=np.float64)
            i_time = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            time = TIMES[i_time]
            # read image:
            arr = np.frombuffer(f.read(width * height * 2), dtype=np.uint16)
            arrs[i] = arr.reshape(width, height)
    #     last row is all zeros in all imgs
    #         print arr[:,:-1]

#             arrs.append(arr)
            labels.append({'exposure time[s]': time,
                           'current[A]': current,
                           'voltage[V]': voltage})
        return arrs, labels


if __name__ == '__main__':
    import pylab as plt
    p = 'YOUR_PATH.elbin'

    if p != 'YOUR_PATH.elbin':
        imgs, labels = elbin(p)

        for n, (img, label) in enumerate(zip(imgs, labels)):
            print(labels)
            plt.figure(n)
            plt.imshow(imgs[n])

        plt.show()
