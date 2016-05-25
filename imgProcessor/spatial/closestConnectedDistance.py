import cv2
import numpy as np
from numba import jit

from imgProcessor.interpolate.interpolate2dStructuredIDW import interpolate2dStructuredIDW


DIRECT_NEIGHBOURS = (-1,0),(1,0), (0,1), (0,-1)



def closestConnectedDistance(target, walls=None, 
                             max_len_border_line=500,
                             max_n_path=100,
                             concentrate_every_n_pixel=1):
    '''
    returns an array with contains the closest distance from every pixel
    the next position where target == 1

    [walls] binary 2darray - e.g. walls in a labyrinth that have to be surrounded in order to get to the target
     
    [target] binary 2darray - positions given by 1
        
    [concentrate_every_n_pixel] often the distance of neighbour pixels is similar
            to speed up calculation set this value to e.g. 3 to calculate only 
            the distance for every 3. pixel and interpolate in between
    
        recommended are values up to 3-5
    
    [max_len_border_line]
    this function calculates distances travelled using region growth
    e.g.
    
    0123
    1123
    2223
    3333
    
    the last steps (e.g. for all steps 3 border_line=7) are stored in an array of limited
    length defined in 'max_len_border_line'
    
    [max_n_path]
    how many paths are possible between every pixel and the target
    only needed if fast==False
    '''
    c = concentrate_every_n_pixel    
    assert c >=1
    if walls is None:
        walls = np.zeros_like(target, dtype=bool)
    s = target.shape
    dt = np.uint16
    if max(target.shape) < 200:
        dt = np.uint8
    out = np.zeros( (s[0]/c,s[1]/c), dtype=dt)
    #temporary arrays:
    growth = np.zeros_like(target, dtype=dt)
    res = np.empty(shape=3,dtype=dt)

    steps = np.empty(shape=(max_len_border_line,2),dtype=dt)
    new_steps = np.empty(shape=(max_len_border_line,2),dtype=dt)

    #run calculation:
    _calc(growth, out, walls, target, steps, new_steps, 
          res, concentrate_every_n_pixel)
    
    if c > 1:
        #if concentrate_every_n_pixel > 1
        #the resized output array
        #will have wrong values close to the wall
        #therefore substitute all wall value (-1)
        #with an average of their closest neighbours
        interpolate2dStructuredIDW(out, out==0)
        out = cv2.resize(out, s[::-1])
        out[walls]=0
    return out
    

@jit(nopython=True)
def _calc(growth, dists, walls, target, steps, new_steps, res, 
          concentrate_every_n_pixel):
    s0,s1 = walls.shape

    #for every pixel:
    for ii,i in enumerate(xrange(0,s0,concentrate_every_n_pixel)):
        for jj,j in enumerate(xrange(0,s1,concentrate_every_n_pixel)): 
            if walls[i,j]:
                continue
            #get the number of steps needed:
            _grow(growth, walls, target, i,j, steps, new_steps, res)
            
            if res[0]:
                #approximate travelled distance as number of growth steps taken
                #this is fast but will return a too short path
                dists[ii,jj] = growth[res[1],res[2]] + 1
               
    
@jit(nopython=True)
def _grow(growth, walls, target, i,j, steps, new_steps, res):
    '''
    fills [res] with [distance to next position where target == 1, 
                      x coord.,
                      y coord. of that position in target]
     using region growth
    
    i,j -> pixel position
     growth -> a work array, needed to measure the distance
     steps, new_steps -> current and last positions of the region growth steps
        using this instead of looking for the right step position in [growth]
        should speed up the process
    '''

    #clean array:
    growth[:] = 0

    if target[i,j]:
        #pixel is in target
        res[0]=1
        res[1]=i
        res[2]=j
        return
    
    step = 1
    s0,s1 = growth.shape
    step_len = 1
    new_step_ind = 0
    
    steps[new_step_ind,0] = i
    steps[new_step_ind,1] = j
    growth[i,j]=1

    while True:
        for n in xrange(step_len):
            i,j = steps[n]
            for ii,jj in  DIRECT_NEIGHBOURS:
                    pi = i+ii
                    pj = j+jj
                    
                    #if in image:
                    if 0<=pi<s0 and 0<=pj<s1:
                        #is growth array is empty and there are no walls:
                            #fill growth with current step
                        if growth[pi,pj] == 0 and not walls[pi,pj]:
                            growth[pi,pj] = step
                            if target[pi,pj]:
                                #found destination
                                res[0]=1
                                res[1]=pi
                                res[2]=pj
                                return
                            
                            new_steps[new_step_ind,0]=pi
                            new_steps[new_step_ind,1]=pj                            
                            new_step_ind += 1

        if new_step_ind == 0:
            #couldn't populate any more because growth is full
                #and all possible steps are gone
            res[0]=0
            return
        
        step += 1
        steps,new_steps  = new_steps, steps
        step_len = new_step_ind
        new_step_ind = 0
      

  
if __name__ == '__main__':
    import pylab as plt

    
    size = (50,60)

    walls = np.zeros(size, dtype=bool)
    #add walls
    walls[5:-5, 35]=1
    walls[35:-5, 20]=1
    walls[-5, 20:34]=1
    walls[35, 20:35]=1

    #draw diag:
    for i in xrange(10,20):
        walls[i,i]=1
    
    target = np.zeros(size, dtype=bool)
    target[:,2]=1
    target[:,-2]=1

    dists = closestConnectedDistance(target, walls, 
                                     concentrate_every_n_pixel=1)

    dists[target] = dists.max()
    plt.figure('Walls=Blue, Target=Red')
    
    plt.imshow(dists, interpolation='none')
    plt.colorbar()
  
    plt.show()
    

