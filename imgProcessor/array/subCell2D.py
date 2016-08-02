import numpy as np

def subCell2DGenerator(arr, (g0,g1)):
    '''
    generator to access evenly sized sub-cells in a 2d array
    '''
    x,y=0,0
    s0,s1 = arr.shape[:2]
    d0,d1 = int(round(float(s0)/g0)),int(round(float(s1)/g1))
    y1 = d0
    for i in xrange(g0):
        for j in xrange(g1):
            x1 = x+d1
            yield i,j, arr[y:y1,x:x1]
            x = x1
        y = y1
        y1 = y+d0
        x = 0


def subCell2DFnArray(arr, fn, cells, dtype=None):
    '''
    Return array where every cell is the output of a given cell function
    mx = subCell2DFnArray(myArray, np.max, (10,6) )
    --> here mx is a 2d array containing all cell maxima
    '''
    out = np.empty(shape=cells, dtype=dtype)
    for i,j,c in subCell2DGenerator(arr, cells):
        out[i,j] = fn(c)
    return out