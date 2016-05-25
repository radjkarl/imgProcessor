import cv2
from numpy.linalg import inv
import numpy as np


def imgPointToWorldCoord((ix, iy), rvec, tvec, cameraMatrix, zconst=0):
    '''
    @returns 3d object coords 

    :param (ix,iy): list of 2d points (x,y) in image
    :param zconst: height above image plane (if 0, than on image plane)
    
    http://answers.opencv.org/question/62779/image-coordinate-to-world-coordinate-opencv/
    and
    http://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    '''

    uvPoint = np.array([ix.ravel(),iy.ravel(),np.ones(shape=ix.size)]).reshape(3,ix.size)

    R = cv2.Rodrigues(rvec)[0]
    
    iR = inv(R)
    iC = inv(cameraMatrix)

    t = iR.dot(iC).dot(uvPoint)
    t2 = iR.dot(tvec)
    s = (zconst + t2[2]) / t[2]
    
    objP = (iR.dot (s* iC.dot(uvPoint) - tvec))
    return objP



if __name__ == '__main__':
    from imgProcessor.genericCameraMatrix import genericCameraMatrix    
    
    #assume:
    cameraMatrix = genericCameraMatrix((600,800))
    distCoeffs=0

    #taken from http://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    #img points are green dots in the picture
    imagePoints = np.array([(271.,109.),
                            (65.,208.),
                            (334.,459.),
                            (600.,225.)])
    
    #object points are measured in millimeters because calibration is done in mm also
    objectPoints = np.array([(0., 0., 0.), 
                             (-511.,2181.,0.), 
                             (-3574.,2354.,0.),
                             (-3400.,0.,0.)])
    
    #estimate pose:
    ret, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

    #test via finding image point from obj points:
    imgPoints2 = cv2.projectPoints(np.array([objectPoints]), rvec, tvec, cameraMatrix, distCoeffs=0)[0]
    #should be close to imagePoints:
    assert np.allclose(imagePoints,imgPoints2,rtol=10)

    ###
    #find object points for that image point:
    ix = np.array([363.0])
    iy = np.array([222])
    zconst=285
    
    objectPoints2 = imgPointToWorldCoord((ix,iy), rvec, tvec, cameraMatrix, zconst)
    #should be close to  [-2629.5, 1272.6, 285.]
    assert np.allclose(objectPoints2,[-2629.5, 1272.6, 285.],rtol=50)


    
    