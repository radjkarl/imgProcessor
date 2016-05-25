import numpy as np


def gaussian2d((x,y), sx, sy, mx=0, my=0, rho=0):
    '''
    see http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # probability density function of a vector [x,y]
    sx,sy -> sigma (standard deviation)
    mx,my: mue (mean position)
    rho: correlation between x and y
    '''
    return ( 
        1/(2*np.pi*sx*sy*(1-(rho**2))**0.5) *
         np.exp( (-1/(2*(1-rho**2))) *
                 (
                    ( (x-mx)**2/sx**2 )
                  + ( (y-my)**2/sy**2 )
                  - ( ( 2*rho*(x-mx)*(y-my)) / (sx*sy) )
                  )
                )
         )