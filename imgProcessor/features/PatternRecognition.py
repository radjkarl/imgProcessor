from __future__ import division
from __future__ import print_function

import cv2
# cv2.ocl.setUseOpenCL(False)

import numpy as np
# from skimage.transform import resize

from imgProcessor.imgSignal import scaleSignalCutParams
from imgProcessor.transformations import toUIntArray


class PatternRecognition(object):

    def __init__(self, image,
                 maxImageSize=1000,
                 minInlierRatio=0.15, minInliers=25,
                 fast=False):
        '''
        maxImageSize -> limit image size to speed up process, set to False to deactivate

        minInlierRatio --> [e.g:0.2] -> min 20% inlier need to be matched, else: raise Error

        '''
        self.signal_ranges = []
        self.maxImageSize = maxImageSize
        self.minInlierRatio = minInlierRatio
        self.minInliers = minInliers
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
                # scoreType=cv2.ORB_FAST_SCORE
            )

        # removed because of license issues:
        # cv2.xfeatures2d.SIFT_create()
        f, d = self.detector.detectAndCompute(self.base8bit, None)
        self.base_features, self.base_descs = f, d  # .astype(np.float32)

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
            s0 = int(s0)
            s1 = int(s1)

            # TODO: find out why bad fit if not transformed to np.float32
            image = cv2.resize(image, (s1, s0)).astype(np.float32)

        img8bit = self._scaleTo8bit(image)

        if img8bit.ndim == 3:  # multi channel img like rgb
            img8bit = cv2.cvtColor(img8bit, cv2.COLOR_BGR2GRAY)
        return img8bit
#         return cv2.GaussianBlur(img8bit, (self.fineKernelSize,
#                                           self.fineKernelSize), 0)

    def _scaleTo8bit(self, img):
        '''
        The pattern comparator need images to be 8 bit
        -> find the range of the signal and scale the image
        '''
        r = scaleSignalCutParams(img, 0.02)  # , nSigma=3)
        self.signal_ranges.append(r)
        return toUIntArray(img, dtype=np.uint8, range=r)

    def _filterMatches(self, matches, ratio=0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])
        return filtered_matches

    def invertHomography(self, H):
        return np.linalg.inv(H)

    def findHomography(self, img, drawMatches=False):
        '''
        Find homography of the image through pattern
        comparison with the base image
        '''
        print("\t Finding points...")
        # Find points in the next frame
        img = self._prepareImage(img)
        features, descs = self.detector.detectAndCompute(img, None)

        ######################
        # TODO: CURRENTLY BROKEN IN OPENCV3.1 - WAITNG FOR NEW RELEASE 3.2
#         matches = self.matcher.knnMatch(descs,#.astype(np.float32),
#                                         self.base_descs,
#                                         k=3)
#         print("\t Match Count: ", len(matches))
#         matches_subset = self._filterMatches(matches)

        # its working alternative (for now):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_subset = bf.match(descs, self.base_descs)

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
                                       cv2.RANSAC,  # method
                                       5.0  # max reprojection error (1...10)
                                       )
        if status is None:
            raise Exception('no homography found')
        else:
            inliers = np.sum(status)
            print('%d / %d  inliers/matched' % (inliers, len(status)))
            inlierRatio = inliers / len(status)
            if self.minInlierRatio > inlierRatio or inliers < self.minInliers:
                raise Exception('bad fit!')

        # scale with _fH, if image was resized
        # see
        # http://answers.opencv.org/question/26173/the-relationship-between-homography-matrix-and-scaling-images/
        s = np.eye(3, 3)
        s[0, 0] = 1 / self._fH
        s[1, 1] = 1 / self._fH
        H = s.dot(H).dot(np.linalg.inv(s))

        if drawMatches:
            #             s0,s1 = img.shape
            #             out = np.empty(shape=(s0,s1,3), dtype=np.uint8)
            img = draw_matches(self.base8bit, self.base_features, img, features,
                               matches_subset[:20],  # None,#out,
                               # flags=2
                               thickness=5
                               )

        return (H, inliers, inlierRatio, averagePointDistance,
                img, features,
                descs, len(matches_subset))


# TAKEN FROM https://gist.github.com/CannedYerins/11be0c50c4f78cad9549
# allows to set radius and line thickness.
# remove or more to other module later -> only needed to visualization
# and debugging
def draw_matches(img1, kp1, img2, kp2, matches, color=None, thickness=2, r=15):
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[
                     1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (
            max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]
        :img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second
    # image appropriately.
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0, 256, 3) if len(
                img1.shape) == 3 else np.random.randint(0, 256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(
            int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img


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
