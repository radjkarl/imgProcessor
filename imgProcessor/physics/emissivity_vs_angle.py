'''
relative emissivity over variable angle of view
'''

import numpy as np


def EL_Si_module():
    '''
    calculated of nanmedian(persp-corrected EL module/reference module)
    published in ########
    '''
#     #angles as seupt:
#     angles = np.array([0,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
#                        55, 60, 65, 70, 75, 80, 85], dtype=np.float32)
#     #actual angles considering distance between camera and device:
# #     angles = np.array([ 0,11.4967447384,13.4871127692, 16.7292217609,20.736729996,25.1040773237,29.5787204549, 34.2304538602,38.9905808891,43.8419066693,48.7688750265,53.7897698954,58.8285620176 ,63.9331370804,69.121827123,74.3321311931,79.5246356463,84.717380152  ], dtype=np.float32) 
#     vals = np.array([1,  1.0021438598632813, 1.0044595003128052, 
#                      1.0017602443695068, 0.99988853931427002, 
#                      0.99665242433547974, 0.99339976906776428, 
#                      0.9893726110458374, 0.98351168632507324, 
#                      0.97270435094833374, 0.96261692047119141,
#                      0.94098988175392151, 0.9121357798576355, 
#                      0.86570969223976135, 0.77970236539840698, 
#                      0.65183627605438232, 0.45379666984081268, 
#                      0.16491696238517761], dtype=np.float32)
    
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
    a,v = EL_Si_module()
    plt.plot(a,v)
    plt.show()
      
      