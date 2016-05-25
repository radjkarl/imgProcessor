import numpy as np
from scipy.spatial import ConvexHull

BL_ANGLE = 2.356194490192345#=135degree


def sortCorners(corners):
    '''
    sort the corners of a given quadrilateral of the type
    corners : [ [xi,yi],... ]
    
    to an anti-clockwise order starting with the bottom left corner
    
    or (if plotted as image where y increases to the bottom):
    clockwise, starting top left
    '''
    corners = np.asarray(corners)
    #bring edges in order:
    corners = corners[ConvexHull(corners).vertices]
    #find the edge with the right angle to the quad middle:
    mn = corners.mean(axis=0)
    d = (corners-mn)
    ascent = np.arctan2(d[:,1],d[:,0])
    bl = np.abs(BL_ANGLE+ascent).argmin()
    #build a index list starting with bl:
    i = range(bl,4)
    i.extend(range(0,bl))
    return corners[i]

if __name__ == '__main__':
    import pylab as plt
    
    corners = np.array(((0,10),(-20,20),(20,50),(-30,70)))
    sortedC = sortCorners(corners)
    
    plt.plot(corners[:,0],corners[:,1],label='unsorted')
    plt.plot(sortedC[:,0],sortedC[:,1],label='sorted')
    plt.legend()
    plt.show()
    