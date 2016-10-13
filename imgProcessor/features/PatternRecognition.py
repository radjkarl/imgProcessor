from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from skimage.transform import resize

from imgProcessor.imgSignal import signalRange


class PatternRecognition(object):

    def __init__(self, image, fineKernelSize=3, maxImageSize=1000):
        '''
        maxImageSize -> limit image size to speed up process, set to False to deactivate
        '''
        self.fineKernelSize = fineKernelSize
        self.signal_ranges = []
        self.maxImageSize = maxImageSize
        self._fH = None  # homography factor, if image was resized

        self.base8bit = self._prepareImage(image)

        # Parameters for nearest-neighbour matching
        flann_params = dict(algorithm=1, trees=2)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})

        # PATTERN DETECTOR:
        self.detector = cv2.BRISK_create()
        # removed because of license issues:
        # cv2.xfeatures2d.SIFT_create()
        f, d = self.detector.detectAndCompute(self.base8bit, None)
        self.base_features, self.base_descs = f, d

    def _prepareImage(self, image):
        # resize is image too big:
        m = self.maxImageSize
        if m:
            s0, s1 = image.shape[:2]
            if self._fH is None:
                m = float(m)

                if max((s0, s1)) > m:
                    if s0 > s1:
                        self._fH = m // s0
                        s1 *= self._fH
                        s0 = m
                    else:
                        self._fH = m // s1
                        s0 *= m // s1
                        s1 = m
                else:
                    self._fH = 1
            else:
                s0 *= self._fH
                s1 *= self._fH

            image = resize(image, (int(round(s0)), int(round(s1))),
                           preserve_range=True)

        img8bit = self._scaleTo8bit(image)

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

#
#     #only for opencv2.4 because that method is built into 3.0  - remove later
#     def _drawMatches(self, img1, kp1, img2, kp2, matches):
#         """
#         ##TAKEN FROM 
#         ##http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
#         #DOESNT WORK FOR knn matcher
#
#         My own implementation of cv2.drawMatches as OpenCV 2.4.9
#         does not have this function available but it's supported in
#         OpenCV 3.0.0
#
#         This function takes in two images with their associated
#         keypoints, as well as a list of DMatch data structure (matches)
#         that contains which keypoints matched in which images.
#
#         An image will be produced where a montage is shown with
#         the first image followed by the second image beside it.
#
#         Keypoints are delineated with circles, while lines are connected
#         between matching keypoints.
#
#         img1,img2 - Grayscale images
#         kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
#                   detection algorithms
#         matches - A list of matches of corresponding keypoints through any
#                   OpenCV keypoint matching algorithm
#         """
#
#         # Create a new output image that concatenates the two images together
#         # (a.k.a) a montage
#         rows1 = img1.shape[0]
#         cols1 = img1.shape[1]
#         rows2 = img2.shape[0]
#         cols2 = img2.shape[1]
#
#         out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
#
#         # Place the first image to the left
#         out[:rows1,:cols1] = np.dstack([img1, img1, img1])
#
#         # Place the next image to the right of it
#         out[:rows2,cols1:] = np.dstack([img2, img2, img2])
#
#         # For each pair of points we have between both images
#         # draw circles, then connect a line between them
#         for mat in matches:
#
#             # Get the matching keypoints for each of the images
#             print(mat, dir(mat))
#
#             img1_idx = mat.queryIdx
#             img2_idx = mat.trainIdx
#
#             # x - columns
#             # y - rows
#             (x1,y1) = kp1[img1_idx].pt
#             (x2,y2) = kp2[img2_idx].pt
#
#             # Draw a small circle at both co-ordinates
#             # radius 4
#             # colour blue
#             # thickness = 1
#             cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
#             cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
#
#             # Draw a line in between the two points
#             # thickness = 1
#             # colour blue
#             cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
#
#         # Show the image
#         cv2.namedWindow('Matched Features',
#                          cv2.WINDOW_NORMAL
#                         #old: cv2.cv.CV_WINDOW_NORMAL
#                         )
#
#         cv2.imshow('Matched Features', out)
#         cv2.waitKey(0)
#         cv2.destroyWindow('Matched Features')
#
#         # Also return the image if you'd like a copy
#         return out

    # alternative method - might remove later
    def _findHomography_knnFlann(self, img):
        img = self._prepareImage(img)

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()  # old: cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.base8bit, None)
        kp2, des2 = sift.detectAndCompute(img, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

        inliers = np.sum(status)
        inlierRatio = inliers / len(status)

        return (H, inliers, inlierRatio)

    def findHomography(self, img):
        '''
        Find homography of the image through pattern
        comparison with the base image
        '''
        print("\t Finding points...")
        # Find points in the next frame
        img = self._prepareImage(img)

        features, descs = self.detector.detectAndCompute(img, None)

        matches = self.matcher.knnMatch(descs.astype(np.float32),
                                        self.base_descs.astype(np.float32),
                                        k=2)
        print("\t Match Count: ", len(matches))
        matches_subset = self._filterMatches(matches)
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

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

        inliers = np.sum(status)
        inlierRatio = inliers / len(status)
        # scale with _fH, if image was resized
        # see
        # http://answers.opencv.org/question/26173/the-relationship-between-homography-matrix-and-scaling-images/
        s = np.eye(3, 3)
        s[0, 0] = 1 / self._fH
        s[1, 1] = 1 / self._fH
        H = s.dot(H).dot(np.linalg.inv(s))

        return (H, inliers, inlierRatio, averagePointDistance,
                img, features,
                descs, matches_subset)

    # alternative method - might remove later
    def _findHomography_BR(self, img):
        '''
        Find homography of the image through pattern
        comparison with the base image
        '''
        print("\t Finding points...")
        # Find points in the next frame
        img = self._prepareImage(img)
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()  # old: cv2.SIFT()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.base8bit, None)
        kp2, des2 = sift.detectAndCompute(img, None)
        # Create matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Do matching
        matches = bf.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

        inliers = np.sum(status)
        inlierRatio = inliers / len(status)
        return (H, inliers, inlierRatio)


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
