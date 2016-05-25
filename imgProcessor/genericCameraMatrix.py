import numpy as np


def genericCameraMatrix(shape, angularField=60): 
    '''
    Return a generic camera matrix
    [[fx, 0, cx],
    [ 0, fy, cy],
    [ 0, 0,   1]]
    for a given image shape
    '''
    #http://nghiaho.com/?page_id=576
    #assume that the optical centre is in the middle:
    cy = int(shape[0] / 2.0)
    cx = int(shape[1] /2.0)
    
    #assume that the FOV is 60 DEG (webcam)
    fx = fy = cx/np.tan(angularField/2 * np.pi / 180) #camera focal length
    # see http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0,   1]
                     ], dtype=np.float32)
    