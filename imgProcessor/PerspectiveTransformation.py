import os
import cv2
import math
import numpy as np
from imgProcessor.PatternRecognition import PatternRecognition



class PerspectiveTransformation(object):
    '''
    fit or add an image to the first image of this display 
    using perspective transformations
    '''
    def __init__(self, img):
        '''
        @param img -> reference image
        '''
        self.img_orig = img
        self._firstTime = True
        self.pattern = PatternRecognition(img)
  

    def fitImg(self, img_rgb):
        '''
        fit perspective and size of the input image to the base image
        '''
        H = self.pattern.findHomography(img_rgb)[0]
        H_inv = self.pattern.invertHomography(H)
        s = self.img_orig.shape
        warped = cv2.warpPerspective(img_rgb, H_inv, (s[1],s[0]) )
        return warped


    def addImg(self, img, overlap=None, direction='bottom'):
        '''
        '''
        assert direction == 'bottom','only direction=bottom implemented by now'
        
        #CUT IMAGE TO ONLY COMPARE POINTS AT OVERLAP:
        if overlap is not None:
            #only direction bottom for now...
            s = self.img_orig.shape
            oimgcut = self.img_orig[s[0]-overlap:,:]
            imgcut = img[:overlap,:]
        else:
            oimgcut = self.img_orig
            imgcut = img
        
        #PATTERN COMPARISON:
        if not self._firstTime or overlap is not None:
            self.pattern = PatternRecognition(oimgcut)   
        (H, inlierRatio) = self.pattern.findHomography(imgcut)[0:2]
        H_inv = self.pattern.invertHomography(H)

        #STITCH:
        self.img_orig = self._stitchImg(H_inv, inlierRatio, img, overlap)
        self._firstTime = False
        return self.img_orig


    def addDir(self, image_dir, img_filter=None):  
        '''
        @param image_dir -> 'directory' containing all images
        @param img_filter -> 'JPG'; None->Take all images
        '''
        dir_list = []
        try:
            dir_list = os.listdir(image_dir)
            if img_filter:
                # remove all files that doen't end with .[image_filter]
                dir_list = filter(lambda x: x.find(img_filter) > -1, dir_list)
            try: #remove Thumbs.db, is existent (windows only)
                dir_list.remove('Thumbs.db')
            except ValueError:
                pass
        except:
            raise IOError("Unable to open directory: %s" % image_dir)
        dir_list = map(lambda x: os.path.join(image_dir, x), dir_list)
        dir_list = filter(lambda x: x != image_dir, dir_list)
        return self._stitchDirRecursive(self.base_img_rgb, dir_list, 0)        
        

    def filterMatches(self, matches, ratio = 0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])
        return filtered_matches
   
    
    def imageDistance(self, matches):
        sumDistance = 0.0
        for match in matches:
            sumDistance += match.distance
        return sumDistance
   
    
    def findDimensions(self, image, homography):
        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)
    
        (y, x) = image.shape[:2]
    
        base_p1[:2] = [0,0]
        base_p2[:2] = [x,0]
        base_p3[:2] = [0,y]
        base_p4[:2] = [x,y]
    
        max_x = None
        max_y = None
        min_x = None
        min_y = None
    
        for pt in [base_p1, base_p2, base_p3, base_p4]:
    
            hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
            hp_arr = np.array(hp, np.float32)
            normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

            if ( max_x == None or normal_pt[0,0] > max_x ):
                max_x = normal_pt[0,0]
            if ( max_y == None or normal_pt[1,0] > max_y ):
                max_y = normal_pt[1,0]
            if ( min_x == None or normal_pt[0,0] < min_x ):
                min_x = normal_pt[0,0]
            if ( min_y == None or normal_pt[1,0] < min_y ):
                min_y = normal_pt[1,0]
    
        min_x = min(0, min_x)
        min_y = min(0, min_y)
    
        return (min_x, min_y, max_x, max_y)

    
    def _stitchDirRecursive(self, dir_list, round=0):
        if ( len(dir_list) < 1 ):
            return self.base_img_rgb
        # Find key points in base image for motion estimation
        self.detector.detectAndCompute(self.base_img, None)
        
        print "Iterating through next images..."
    
        closestImage = None
    
        # TODO: Thread this loop since each iteration is independent
    
        # Find the best next image from the remaining images
        for next_img_path in dir_list:
    
            print "Reading %s..." % next_img_path
    
            next_img_rgb = cv2.imread(next_img_path)

            (H, inlierRatio, averagePointDistance, 
            next_img, next_features, 
            next_descs, matches_subset) = self.pattern.findHomography(next_img_rgb)
            
            # if ( closestImage == None or averagePointDistance < closestImage['dist'] ):
            if ( closestImage == None or inlierRatio > closestImage['inliers'] ):
                closestImage = {}
                closestImage['h'] = H
                closestImage['inliers'] = inlierRatio
                closestImage['dist'] = averagePointDistance
                closestImage['path'] = next_img_path
                closestImage['rgb'] = next_img_rgb
                closestImage['img'] = next_img
                closestImage['feat'] = next_features
                closestImage['desc'] = next_descs
                closestImage['match'] = matches_subset
    
        print "Closest Image: ", closestImage['path']
        print "Closest Image Ratio: ", closestImage['inliers']

        dir_list = filter(lambda x: x != closestImage['path'], dir_list)

        self.base_img_rgb = self._stitchImg(closestImage)
        self.base_img = cv2.GaussianBlur(cv2.cvtColor(self.base_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)

        return self._stitchDirRecursive(dir_list, round+1)


    def _stitchImg(self,H_inv, inliers, img, overlap=0):
        # TODO: use img_orig with can be float array
        # to return stitched results as floatarray of the same kind

        isColor = img.ndim == 3

        if ( inliers > 0.1 ): # and 
    
            #add translation to homography to consider overlap:
            if overlap:
                H_inv[1,2]+=self.img_orig.shape[0]-overlap

            (min_x, min_y, max_x, max_y) = self.findDimensions(img, H_inv)
    
            # Adjust max_x and max_y by base img size
            max_x = max(max_x, self.img_orig.shape[1])
            max_y = max(max_y, self.img_orig.shape[0])
    
            move_h = np.matrix(np.identity(3), np.float32)
    
            if ( min_x < 0 ):
                move_h[0,2] += -min_x
                max_x += -min_x
    
            if ( min_y < 0 ):
                move_h[1,2] += -min_y 
                max_y += -min_y
                
#             print "Homography: \n", H
            print "Inverse Homography: \n", H_inv
            print "Min Points: ", (min_x, min_y)
    
            mod_inv_h = move_h * H_inv

    
            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))
    
            print "New Dimensions: ", (img_w, img_h)
    
            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(self.img_orig, move_h, (img_w, img_h))
            print "Warped base image"
    
            next_img_warp = cv2.warpPerspective(img, mod_inv_h, (img_w, img_h))
            print "Warped next image"
    
            # Put the base image on an enlarged palette
            if isColor:
                enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
            else:
                enlarged_base_img = np.zeros((img_h, img_w), np.uint8)
    
            print "Enlarged Image Shape: ", enlarged_base_img.shape
            print "Base Image Shape: ", self.img_orig.shape
            print "Base Image Warp Shape: ", base_img_warp.shape
    

            # Create a mask from the warped image for constructing masked composite
            if isColor:
                d = np.sum(next_img_warp, axis=-1)
            else:
                d = next_img_warp

            # Now add the warped image
            data_map = d==0
            enlarged_base_img[data_map] = base_img_warp[data_map]
            final_img = enlarged_base_img + next_img_warp
            
            #average overlap:
            if isColor:
                dd = np.sum(base_img_warp, axis=-1)
            else:
                dd = base_img_warp
            mask = np.logical_and(d!=0, dd!=0) 
            av = 0.5*(cv2.subtract(base_img_warp[mask],next_img_warp[mask]))
            if not isColor:
                av = av[:,0]
            final_img[mask] += av

#             final_img = cv2.add(enlarged_base_img, next_img_warp, 
#                 dtype=cv2.CV_8U)

            # Crop off the black edges
#             final_gray = self._rgb2Gray(final_img)
#             _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
#             
            thresh = final_img>0
            if isColor:
                thresh = np.sum(thresh, axis=0)
            
            contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print "Found %d contours..." % (len(contours))
    
            max_area = 0
            best_rect = (0,0,0,0)
    
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                # print "Bounding Rectangle: ", (x,y,w,h)
    
                deltaHeight = h-y
                deltaWidth = w-x
    
                area = deltaHeight * deltaWidth
    
                if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
                    max_area = area
                    best_rect = (x,y,w,h)
    
            if ( max_area > 0 ):
                print "Maximum Contour: ", max_area
                print "Best Rectangle: ", best_rect
    
                final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                        best_rect[0]:best_rect[0]+best_rect[2]]
    
                final_img = final_img_crop
    
            return final_img
    
        else:
            return self.base_img_rgb

    

if __name__ == '__main__':
    import timeit
    from fancytools.os.PathStr import PathStr

    d = PathStr('media')
    i1 = cv2.imread(d.join('peanut_1.jpg'))
    i2 = cv2.imread(d.join('peanut_2.jpg'))

    
    def fn(overlap=None):
        #stitch 2 images taken in 2 different perspectives together:
        p = PerspectiveTransformation(i1) 
        fn.result = p.addImg(i2, overlap=overlap) 
    
    #lets find out which method is faster:
    print( 'time needed without given overlap [s]: ', 
           timeit.timeit(fn, number=1) )
    print( 'time needed with given overlap [s]: ', 
           timeit.timeit(lambda: fn(overlap=150), number=1) )


    cv2.namedWindow("warped")
    cv2.imshow('warped',fn.result)
    cv2.waitKey()