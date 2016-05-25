import numpy as np
from fancytools.math import line



def calcAspectRatioFromCorners(corners, in_plane=False):
    '''
    simple and better alg. than below
    in_plane -> whether object has no tilt, but only rotation and translation
    '''
    
    q =  corners
    l0 = [q[0,0],q[0,1],q[1,0],q[1,1]]
    l1 = [q[0,0],q[0,1],q[-1,0],q[-1,1]]

    l2 = [q[2,0],q[2,1],q[3,0],q[3,1]]
    l3 = [q[2,0],q[2,1],q[1,0],q[1,1]]    

    a1 = line.length(l0) / line.length(l1)                    
    a2 = line.length(l2) / line.length(l3)
    
    if in_plane:
        #take aspect ration from more rectangular corner
        if (    abs( 0.5*np.pi - abs(line.angle2(l0, l1)) )
               < abs( 0.5*np.pi - abs(line.angle2(l2, l3)) ) ):
            return a1
        else:
            return a2
 
    return 0.5*(a1+a2)

##############
#TODO: maybe define sqewness and than decide whether to use easy or complicated alg
#############
# 
# def calcAspectRatioFromCorners_old(corners):
#     '''
#     this code monster is taken from:
#     http://stackoverflow.com/questions/1194352/proportions-of-a-perspective-deformed-rectangle
#     which cites:
#     Zhengyou Zhang , Li-Wei He, "Whiteboard scanning and image enhancement" 
#     http://research.microsoft.com/en-us/um/people/zhang/papers/tr03-39.pdf
#     '''      
#     m1x,m1y = float(corners[0][0]), float(corners[0][1])
#     m2x,m2y = float(corners[1][0]), float(corners[1][1])
#     m4x,m4y = float(corners[2][0]), float(corners[2][1])
#     m3x,m3y = float(corners[3][0]), float(corners[3][1])
#     
#     try:
#         res = np.sqrt(((((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y -
#         m1y*m4x)/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x) -
#         1)*(((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x)/((m2y -
#         m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) - 1)*(((m1y -
#         m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x)*m2y/((m2y - m4y)*m3x
#         - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) - m1y)**2/((((m1y - m4y)*m2x -
#         (m1x - m4x)*m2y + m1x*m4y - m1y*m4x)*m3y/((m3y - m4y)*m2x - (m3x -
#         m4x)*m2y + m3x*m4y - m3y*m4x) - m1y)*(((m1y - m4y)*m3x - (m1x -
#         m4x)*m3y + m1x*m4y - m1y*m4x)*m2y/((m2y - m4y)*m3x - (m2x - m4x)*m3y
#         + m2x*m4y - m2y*m4x) - m1y) + (((m1y - m4y)*m2x - (m1x - m4x)*m2y +
#         m1x*m4y - m1y*m4x)*m3x/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y
#         - m3y*m4x) - m1x)*(((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y -
#         m1y*m4x)*m2x/((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x)
#         - m1x)) + (((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y -
#         m1y*m4x)/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x) -
#         1)*(((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x)/((m2y -
#         m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) - 1)*(((m1y -
#         m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x)*m2x/((m2y - m4y)*m3x
#         - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) - m1x)**2/((((m1y - m4y)*m2x -
#         (m1x - m4x)*m2y + m1x*m4y - m1y*m4x)*m3y/((m3y - m4y)*m2x - (m3x -
#         m4x)*m2y + m3x*m4y - m3y*m4x) - m1y)*(((m1y - m4y)*m3x - (m1x -
#         m4x)*m3y + m1x*m4y - m1y*m4x)*m2y/((m2y - m4y)*m3x - (m2x - m4x)*m3y
#         + m2x*m4y - m2y*m4x) - m1y) + (((m1y - m4y)*m2x - (m1x - m4x)*m2y +
#         m1x*m4y - m1y*m4x)*m3x/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y
#         - m3y*m4x) - m1x)*(((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y -
#         m1y*m4x)*m2x/((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x)
#         - m1x)) - (((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y -
#         m1y*m4x)/((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) -
#         1)**2)/((((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y -
#         m1y*m4x)/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x) -
#         1)*(((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x)/((m2y -
#         m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) - 1)*(((m1y -
#         m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y - m1y*m4x)*m3y/((m3y - m4y)*m2x
#         - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x) - m1y)**2/((((m1y - m4y)*m2x -
#         (m1x - m4x)*m2y + m1x*m4y - m1y*m4x)*m3y/((m3y - m4y)*m2x - (m3x -
#         m4x)*m2y + m3x*m4y - m3y*m4x) - m1y)*(((m1y - m4y)*m3x - (m1x -
#         m4x)*m3y + m1x*m4y - m1y*m4x)*m2y/((m2y - m4y)*m3x - (m2x - m4x)*m3y
#         + m2x*m4y - m2y*m4x) - m1y) + (((m1y - m4y)*m2x - (m1x - m4x)*m2y +
#         m1x*m4y - m1y*m4x)*m3x/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y
#         - m3y*m4x) - m1x)*(((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y -
#         m1y*m4x)*m2x/((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x)
#         - m1x)) + (((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y -
#         m1y*m4x)/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x) -
#         1)*(((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x)/((m2y -
#         m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) - 1)*(((m1y -
#         m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y - m1y*m4x)*m3x/((m3y - m4y)*m2x
#         - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x) - m1x)**2/((((m1y - m4y)*m2x -
#         (m1x - m4x)*m2y + m1x*m4y - m1y*m4x)*m3y/((m3y - m4y)*m2x - (m3x -
#         m4x)*m2y + m3x*m4y - m3y*m4x) - m1y)*(((m1y - m4y)*m3x - (m1x -
#         m4x)*m3y + m1x*m4y - m1y*m4x)*m2y/((m2y - m4y)*m3x - (m2x - m4x)*m3y
#         + m2x*m4y - m2y*m4x) - m1y) + (((m1y - m4y)*m2x - (m1x - m4x)*m2y +
#         m1x*m4y - m1y*m4x)*m3x/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y
#         - m3y*m4x) - m1x)*(((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y -
#         m1y*m4x)*m2x/((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x)
#         - m1x)) - (((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y -
#         m1y*m4x)/((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x) -
#         1)**2))
#         
#         if res == 0 or np.isnan(res):
#             raise ZeroDivisionError()
#         return res 
#     except ZeroDivisionError:
#         #code doesnt work if quad is parallelogram
#         #print "couln't determine aspect ratio"
#         #following is stable but not accurate:
#         l14 = ((m4x-m1x)**2+(m4y-m1y)**2)**0.5
#         l12 = ((m2x-m1x)**2+(m2y-m1y)**2)**0.5
#         return l14/l12