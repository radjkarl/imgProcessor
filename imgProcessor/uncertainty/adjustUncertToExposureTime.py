import numpy as np
from scipy.interpolate import UnivariateSpline


def _fitfn(x, a, b, start, end):
    #uncertainty factor is an extended decay function
    return (start - end) * (np.exp(-a * x**b)) + end


def adjustUncertToExposureTime(facExpTime, uncertMap, evtLenMap):
    '''
    Adjust image uncertainty (measured at exposure time t0)
    to new exposure time
    
    facExpTime --> new exp.time / reference exp.time =(t/t0)
    uncertMap --> 2d array mapping image uncertainty
    
    evtLen --> 2d array mapping event duration within image [sec]
                event duration is relative to exposure time
                e.g. duration = 2 means event is 2x longer than 
                exposure time
    
    More information can be found at ...
    ----
    K.Bedrich: Quantitative Electroluminescence Imaging, PhD Thesis, 2017
    Subsection 5.1.4.3: Exposure Time Dependency
    ----
    '''

    #fit parameters, obtained from ####[simulateUncertDependencyOnExpTime]
    params =  np.array( 
        #a                 facExpTime        f_0             f_inf         
     [[  2.63017121e+00,   3.05873627e-01,   1.00000000e+01, 2.78233309e-01],
      [  2.26467931e+00,   2.86206621e-01,   8.01396977e+00, 2.04089232e-01],
      [  1.27361168e+00,   5.18377189e-01,   3.04180084e+00, 2.61396338e-01],
      [  7.34546040e-01,   7.34549823e-01,   1.86507345e+00, 2.77563156e-01],
      [  3.82715618e-01,   9.32410141e-01,   1.34510254e+00, 2.91228149e-01],
      [  1.71166071e-01,   1.14092885e+00,   1.11243702e+00, 3.07947386e-01],
      [  6.13455410e-02,   1.43802520e+00,   1.02995065e+00, 3.93920802e-01],
      [  1.65383071e-02,   1.75605076e+00,   1.00859395e+00, 5.02132321e-01],
      [  4.55800114e-03,   1.99855711e+00,   9.98819118e-01, 5.99572776e-01]])
    
    #event duration relative to exposure time:(1/16...16)
    dur = np.array([  0.0625,   0.125 ,   0.25  ,   
                      0.5   ,   1.    ,   2.    ,
                      4.    ,   8.    ,   16.    ])
    #get factors from interpolation:
    a = UnivariateSpline(dur, params[:, 0], k=3, s=0)
    b = UnivariateSpline(dur, params[:, 1], k=3, s=0)
    start = UnivariateSpline(dur, params[:, 2], k=3, s=0)
    end = UnivariateSpline(dur, params[:, 3], k=3, s=0)
    p0 = a(evtLenMap), b(evtLenMap), start(evtLenMap), end(evtLenMap)
    #uncertainty for new exposure time:
    out = uncertMap * _fitfn(facExpTime, *p0)
    # everywhere where there ARE NO EVENTS --> scale uncert. as if would
    # be normal distributed:
    i = evtLenMap == 0
    out[i] = uncertMap[i] * (1 / facExpTime)**0.5
    return out


if __name__ == '__main__':
    import sys
    import pylab as plt
    
    s0 = 100
    uncert = 0.15 #uncertainty =15%
    dur = 0 # #for majority in text image there are no fluctuations
    
    #exposure time factors:
    f0 = 0.25 #new exposure time is (1/4) of reference time
    f1 = 2  # ...double...
    f2 = 4  # ... four times ... 
    
    uncert = np.full((s0,s0),fill_value=uncert)
    duration = np.full_like(uncert, fill_value=dur)
    #add fluctuations:
    duration[20:30,20:30]=2 #event length if 2x exposure time
    duration[60:80,60:80]=3  #...3x...
    
    ###
    newUncert = adjustUncertToExposureTime(f0, uncert, duration)
    ###
    if 'no_window' not in sys.argv:
        plt.figure('input')
        plt.imshow(uncert)
        plt.colorbar()

        plt.figure('Event duration')
        plt.imshow(duration)
        plt.colorbar()

        plt.figure('Uncertainty for exposure time=%s*ref. exposure time' %f0)
        plt.imshow(newUncert)
        plt.colorbar()

        plt.show()
