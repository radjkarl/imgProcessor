from __future__ import division
from __future__ import print_function

import cv2
# cv2.ocl.setUseOpenCL(False)

import numpy as np
from skimage.transform import resize

from imgProcessor.imgSignal import signalRange



class PatternRecognition(object):

    def __init__(self, image, fineKernelSize=3, maxImageSize=1000, minInlierRatio=0.15,
                 fast=False):
        '''
        maxImageSize -> limit image size to speed up process, set to False to deactivate
        
        minInlierRatio --> [e.g:0.2] -> min 20% inlier need to be matched, else: raise Error
        
        '''
        self.fineKernelSize = fineKernelSize
        self.signal_ranges = []
        self.maxImageSize = maxImageSize
        self.minInlierRatio = minInlierRatio
        self._fH = None  # homography factor, if image was resized

        self.base8bit = self._prepareImage(image)
        # Parameters for nearest-neighbour matching
#         flann_params = dict(algorithm=1, trees=2)
#         self.matcher = cv2.FlannBasedMatcher(flann_params, {})

        # PATTERN DETECTOR:
#         self.detector = cv2.BRISK_create()

        if fast:
            self.detector = cv2.ORB_create()
        else:
            self.detector = cv2.ORB_create(
                    nfeatures=70000, 
                    #scoreType=cv2.ORB_FAST_SCORE
                    )
        
        
        
        # removed because of license issues:
        # cv2.xfeatures2d.SIFT_create()
        f, d = self.detector.detectAndCompute(self.base8bit, None)
        self.base_features, self.base_descs = f, d#.astype(np.float32)


    def _prepareImage(self, image):
        # resize is image too big:
        m = self.maxImageSize
        if m:
            s0, s1 = image.shape[:2]
            if self._fH is None:
                m = float(m)

                if max((s0, s1)) > m:
                    if s0 > s1:
                        self._fH = m / s0
                        s1 *= self._fH
                        s0 = m
                    else:
                        self._fH = m / s1
                        s0 *= m / s1
                        s1 = m
                else:
                    self._fH = 1
            else:
                s0 *= self._fH
                s1 *= self._fH

            image = resize(image, (int(round(s0)), int(round(s1)) ),
                           preserve_range=True)

        img8bit = self._scaleTo8bit(image)
        if img8bit.ndim == 3:#multi channel img like rgb
            img8bit = cv2.cvtColor(img8bit,cv2.COLOR_BGR2GRAY)

        return cv2.GaussianBlur(img8bit, (self.fineKernelSize,
                                          self.fineKernelSize), 0)


    def _scaleTo8bit(self, img):
        '''
        The pattern comparator need images to be 8 bit
        -> find the range of the signal and scale the image
        '''
        r = signalRange(img, nSigma=3)
        self.signal_ranges.append(r)

        if img.dtype == np.uint8:
            return img
        img = 255 * ((np.asfarray(img) - r[0]) / (r[1] - r[0]))
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        return img

    def _filterMatches(self, matches, ratio=0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])
        return filtered_matches

    def invertHomography(self, H):
        # normalize
        #         H /= H[2,2]
        # invert homography
        return np.linalg.inv(H)


    def findHomography(self, img):
        '''
        Find homography of the image through pattern
        comparison with the base image
        '''
        print("\t Finding points...")
        # Find points in the next frame
        img = self._prepareImage(img)

        features, descs = self.detector.detectAndCompute(img, None)

        ######################
        #TODO: CURRENTLY BROKEN IN OPENCV3.1 - WAITNG FOR NEW RELEASE 3.2
#         matches = self.matcher.knnMatch(descs,#.astype(np.float32),
#                                         self.base_descs,
#                                         k=3)
#         print("\t Match Count: ", len(matches))
#         matches_subset = self._filterMatches(matches)

        #its working alternative (for now):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_subset = bf.match(descs,self.base_descs)
        
 
        ######################
#         matches = bf.knnMatch(descs,self.base_descs, k=2)
#         # Apply ratio test
#         matches_subset = []
#         medDist = np.median([m.distance for m in matches])
#         matches_subset = [m for m in matches if m.distance < medDist]
#         for m in matches:
#             print(m.distance)
#         for m,n in matches:
#             if m.distance < 0.75*n.distance:
#                 matches_subset.append([m])
        
        

        if not len(matches_subset):
            raise Exception('no matches found')
        print("\t Filtered Match Count: ", len(matches_subset))

        distance = sum([m.distance for m in matches_subset])
        print("\t Distance from Key Image: ", distance)

        averagePointDistance = distance / (len(matches_subset))
        print("\t Average Distance: ", averagePointDistance)

        kp1 = []
        kp2 = []

        for match in matches_subset:
            kp1.append(self.base_features[match.trainIdx])
            kp2.append(features[match.queryIdx])

        # /self._fH #scale with _fH, if image was resized

        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])  # /self._fH

        H, status = cv2.findHomography(p1, p2, 
                                       cv2.RANSAC,  #method
                                       5.0#max reprojection error (1...10)
                                       )
        if status is None:
            raise Exception('no homography found')
        else:
            inliers = np.sum(status)
            print('%d / %d  inliers/matched' %(inliers, len(status)))
            inlierRatio = inliers/ len(status)
            if self.minInlierRatio>inlierRatio:
                raise Exception('bad fit!')

        # scale with _fH, if image was resized
        # see
        # http://answers.opencv.org/question/26173/the-relationship-between-homography-matrix-and-scaling-images/
        s = np.eye(3, 3)
        s[0, 0] = 1 / self._fH
        s[1, 1] = 1 / self._fH
        H = s.dot(H).dot(np.linalg.inv(s))

        return (H, inliers, inlierRatio, averagePointDistance,
                img, features,
                descs, len(matches_subset))



if __name__ == '__main__':
    import sys
    from fancytools.os.PathStr import PathStr
    import imgProcessor
    from imgProcessor.imgIO import imread
    import pylab as plt

    # 1. LOAD TEST IMAGE
    path = PathStr(imgProcessor.__file__).dirname().join(
        'media', 'electroluminescence', 'EL_cell_cracked.png')
    orig = imread(path)

    # 2. DISTORT REFERENCE IMAGE RANDOMLY:
    rows, cols = orig.shape
    # rescale
    r0 = 1 + 0.2 * (np.random.rand() - 1)
    r1 = 1 + 0.2 * (np.random.rand() - 1)
    dst = cv2.resize(orig, None, fx=r0, fy=r1, interpolation=cv2.INTER_CUBIC)
    # translate
    M = np.float32([[1, 0, np.random.randint(-20, 20)],
                    [0, 1, np.random.randint(-20, 20)]])
    dst = cv2.warpAffine(dst, M, (cols, rows))
    # rotate:
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                np.random.randint(0, 90), 1)
    dst = cv2.warpAffine(dst, M, (cols, rows))

    # 3. CORRECT DISTORTION:
    pat = PatternRecognition(orig)
    h = pat.findHomography(dst)[0]
    hinv = pat.invertHomography(h)
    corrected = cv2.warpPerspective(dst, hinv, orig.shape[::-1])

    # 4. CALCULATE ERROR:
    diff = orig.astype(int) - corrected
    diff[corrected == 0] = 0

    err = np.abs(diff).mean() / orig.mean()
    print('relative error: {:f} %'.format(err * 100))

    assert err < 0.03  # must be smaller 5%

    if 'no_window' not in sys.argv:
        # PLOT:
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(orig)
        axarr[0, 0].set_title('original')

        im = axarr[0, 1].imshow(dst)
        axarr[0, 1].set_title('randomly distorted')
        plt.colorbar(im, ax=axarr[0, 1])

        axarr[1, 0].imshow(corrected)
        axarr[1, 0].set_title('corrected')

        im = axarr[1, 1].imshow(diff)
        axarr[1, 1].set_title('error')
        plt.colorbar(im, ax=axarr[1, 1])

        plt.show()
