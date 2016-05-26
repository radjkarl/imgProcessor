import cv2
import numpy as np

from imgProcessor.signal import signalRange



class PatternRecognition(object):
    def __init__(self, image, fineKernelSize=3):
        self.fineKernelSize = fineKernelSize
        self.signal_ranges = []
        
        self.base8bit = self._prepareImage(image)

        #PATTERN DETECTOR: Use the SIFT
        self.detector = cv2.SIFT()
            #Parameters for nearest-neighbour matching
        flann_params = dict(algorithm=1, trees=2)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        
        self.base_features, self.base_descs = self.detector.detectAndCompute(self.base8bit, None)


    def _prepareImage(self, image):
        img8bit = self._scaleTo8bit(image)
        return cv2.GaussianBlur(img8bit, (self.fineKernelSize,self.fineKernelSize), 0)


    def _scaleTo8bit(self, img):
        '''
        The pattern comparator need images to be 8 bit
        -> find the range of the signal and scale the image
        '''
        if img.dtype == np.uint8:
            return img
        r = signalRange(img, nSigma=3)
        self.signal_ranges.append(r)

        img = 255 * ((np.asfarray(img) - r[0]) / (r[1] - r[0]) )
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
        #normalize
        H /= H[2,2]
        #invert homography
        return np.linalg.inv(H)


    #only for opencv2.4 because that method is built into 3.0  - remove later    
    def _drawMatches(self, img1, kp1, img2, kp2, matches):
        """
        ##TAKEN FROM http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
        #DOESNT WORK FOR knn matcher
        
        My own implementation of cv2.drawMatches as OpenCV 2.4.9
        does not have this function available but it's supported in
        OpenCV 3.0.0
    
        This function takes in two images with their associated 
        keypoints, as well as a list of DMatch data structure (matches) 
        that contains which keypoints matched in which images.
    
        An image will be produced where a montage is shown with
        the first image followed by the second image beside it.
    
        Keypoints are delineated with circles, while lines are connected
        between matching keypoints.
    
        img1,img2 - Grayscale images
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
                  detection algorithms
        matches - A list of matches of corresponding keypoints through any
                  OpenCV keypoint matching algorithm
        """
    
        # Create a new output image that concatenates the two images together
        # (a.k.a) a montage
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]
    
        out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    
        # Place the first image to the left
        out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    
        # Place the next image to the right of it
        out[:rows2,cols1:] = np.dstack([img2, img2, img2])
    
        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:
    
            # Get the matching keypoints for each of the images
            print mat, dir(mat)
            
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
    
            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt
    
            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
    
            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    
        # Show the image
        cv2.namedWindow('Matched Features', cv2.cv.CV_WINDOW_NORMAL)

        cv2.imshow('Matched Features', out)
        cv2.waitKey(0)
        cv2.destroyWindow('Matched Features')
    
        # Also return the image if you'd like a copy
        return out


    #alternative method - might remove later
    def _findHomography_knnFlann(self, img):
        img = self._prepareImage(img)   

        # Initiate SIFT detector
        sift = cv2.SIFT()
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.base8bit,None)
        kp2, des2 = sift.detectAndCompute(img,None)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des1,des2,k=2)
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))

        inliers = np.sum(status)
        inlierRatio = float(inliers) / float(len(status))
        
        return (H, inliers, inlierRatio)


    def findHomography(self, img):
        '''
        Find homography of the image through pattern 
        comparison with the base image
        '''
        print "\t Finding points..."
        # Find points in the next frame   
        img = self._prepareImage(img)    

        features, descs = self.detector.detectAndCompute(img, None)
        matches = self.matcher.knnMatch(descs, trainDescriptors=self.base_descs, k=2)
        print "\t Match Count: ", len(matches)
        matches_subset = self._filterMatches(matches)
        if not len(matches_subset):
            raise Exception('no matches found')
        print "\t Filtered Match Count: ", len(matches_subset)
        
        distance = sum([m.distance for m in matches_subset])
        print "\t Distance from Key Image: ", distance

        averagePointDistance = distance/float(len(matches_subset))
        print "\t Average Distance: ", averagePointDistance

        kp1 = []
        kp2 = []

        for match in matches_subset:
            kp1.append(self.base_features[match.trainIdx])
            kp2.append(features[match.queryIdx])

        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))

        inliers = np.sum(status)
        inlierRatio = float(inliers) / float(len(status))

        return (H, inliers, inlierRatio, averagePointDistance, 
            img, features, 
            descs, matches_subset)


    #alternative method - might remove later
    def _findHomography_BR(self, img):
        '''
        Find homography of the image through pattern 
        comparison with the base image
        '''
        print "\t Finding points..."
        # Find points in the next frame   
        img = self._prepareImage(img)     
        # Initiate SIFT detector
        sift = cv2.SIFT()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.base8bit,None)
        kp2, des2 = sift.detectAndCompute(img,None)
        # Create matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Do matching
        matches = bf.knnMatch(des1,des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))

        inliers = np.sum(status)
        inlierRatio = float(inliers) / float(len(status))
        return (H, inliers, inlierRatio)
        