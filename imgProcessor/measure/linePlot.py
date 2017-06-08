'''
Created on 6 Oct 2016

@author: elkb4
'''
from scipy.ndimage import map_coordinates
import numpy as np


def linePlot(img, x0, y0, x1, y1, resolution=None, order=3):
    '''
    returns [img] intensity values along line
    defined by [x0, y0, x1, y1]
    
    resolution ... number or data points to evaluate
    order ... interpolation precision
    '''
    if resolution is None:
        resolution = int( ((x1-x0)**2 + (y1-y0)**2 )**0.5 )
    
    if order == 0:
        x = np.linspace(x0, x1, resolution, dtype=int)
        y = np.linspace(y0, y1, resolution, dtype=int)
        return img[y, x]

    x = np.linspace(x0, x1, resolution)
    y = np.linspace(y0, y1, resolution)
    return map_coordinates(img, np.vstack((y,x)), order=order)



if __name__ == '__main__':
    import pylab as plt
    import sys
    #creat synthetic data:
    x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
    z = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)
    #line plot coordinates:
    x0, y0 = 5, 4.5 
    x1, y1 = 60, 75
    ###
    zi = linePlot(z, x0, y0, x1, y1)
    ###
    if 'no_window' not in sys.argv:
        fig, axes = plt.subplots(nrows=2)
        axes[0].imshow(z)
        axes[0].plot([x0, x1], [y0, y1], 'ro-')
        axes[0].axis('image')
        axes[1].plot(zi)
        
        plt.show()