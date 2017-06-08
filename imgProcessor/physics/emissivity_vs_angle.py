'''
relative emissivity over variable angle of view
'''

import numpy as np


def EL_Si_module():
    '''
    returns angular dependent EL emissivity of a PV module
    
    calculated of nanmedian(persp-corrected EL module/reference module)
    
    published in K. Bedrich: Quantitative Electroluminescence Measurement on PV devices
                 PhD Thesis, 2017
    '''
    arr = np.array([
                    [2.5, 1.00281 ],
                    [7.5, 1.00238 ],
                    [12.5, 1.00174],
                    [17.5, 1.00204 ],
                    [22.5, 1.00054 ],
                    [27.5, 0.998255],
                    [32.5, 0.995351],
                    [37.5, 0.991246],
                    [42.5, 0.985304],
                    [47.5, 0.975338],
                    [52.5, 0.960455],
                    [57.5, 0.937544],
                    [62.5, 0.900607],
                    [67.5, 0.844636],
                    [72.5, 0.735028],
                    [77.5, 0.57492 ],
                    [82.5, 0.263214],
                    [87.5, 0.123062]
                    ])

    angles = arr[:,0]
    vals = arr[:,1]

    vals[vals>1]=1
    return angles, vals

    
def TG_glass():
    '''
    reflected temperature for 250DEG Glass
    published in IEC 62446-3 TS: Photovoltaic (PV) systems 
    - Requirements for testing, documentation and maintenance 
    - Part 3: Outdoor infrared thermography of photovoltaic modules 
      and plants p Page 12
    '''
    vals = np.array([(80,0.88),
                     (75,0.88),
                     (70,0.88),
                     (65,0.88),
                     (60,0.88),
                     (55,0.88),
                     (50,0.87),
                     (45,0.86),
                     (40,0.85),
                     (35,0.83),
                     (30,0.80),
                     (25,0.76),
                     (20,0.7),
                     (15,0.60),
                     (10,0.44)])
    #invert angle reference:
    vals[:,0]=90-vals[:,0]
    #make emissivity relative:
    vals[:,1]/=vals[0,1]
    return vals[:,0], vals[:,1]  


if __name__ == '__main__':
    import pylab as plt
    import sys
    a,v = EL_Si_module()
    
    if 'no_window' not in sys.argv:
        plt.plot(a,v)
        plt.show()
    