from imgProcessor.interpolate.polyfit2d import polyfit2dGrid


from imgProcessor.camera.flatField.postprocessing.function import function
from imgProcessor.camera.flatField.postprocessing.polynomial import polynomial

from imgProcessor.equations.angleOfView import angleOfView
from scipy.ndimage.filters import median_filter, gaussian_filter


ppMETHODS = ['KW repair + Gauss', 'KW repair + Median',
             'KW replace', 'AoV replace', 'POLY replace',
             'POLY repair', 'KW repair', 'AoV repair']


def postProcessing(arr, method='KW replace + Gauss', mask=None):
    '''
    Post process measured flat field [arr].
    Depending on the measurement, different
        post processing [method]s are beneficial.
        The available methods are presented in
        ---
        K.Bedrich, M.Bokalic et al.:
        ELECTROLUMINESCENCE IMAGING OF PV DEVICES:
        ADVANCED FLAT FIELD CALIBRATION,2017
        ---

    methods:
        'POLY replace' --> replace [arr] with a 2d polynomial fit
        'KW replace'   -->  ...               a fitted Kang-Weiss function
        'AoV replace'  --> ...                a fitted Angle-of-view function

        'POLY repair' --> same as above but either replacing empty
        'KW repair'       areas of smoothing out high gradient
        'AoV repair'      variations (POLY only)

        'KW repair + Gauss'  --> same as 'KW replace' with additional 
        'KW repair + Median'     Gaussian or Median filter

    mask:
        None/2darray(bool) --> array of same shape ar [arr] indicating
                               invalid or empty positions
    '''
    assert method in ppMETHODS, \
        'post processing method (%s) must be one of %s' % (method, ppMETHODS)

    if method == 'POLY replace':
        return polyfit2dGrid(arr, mask, order=2, replace_all=True)

    elif method == 'KW replace':
        return function(arr, mask, replace_all=True)

    elif method == 'POLY repair':
        return polynomial(arr, mask, replace_all=False)

    elif method == 'KW repair':
        return function(arr, mask, replace_all=False)

    elif method == 'KW repair + Median':
        return median_filter(function(arr, mask, replace_all=False),
                             min(method.shape) // 20)
    elif method == 'KW repair + Gauss':
        return gaussian_filter(function(arr, mask, replace_all=False),
                               min(arr.shape) // 20)

    elif method == 'AoV repair':
        return function(arr, mask, fn=lambda XY, a:
                        angleOfView(XY, method.shape, a=a), guess=(0.01),
                        down_scale_factor=1)

    elif method == 'AoV replace':
        return function(arr, mask, fn=lambda XY, a:
                        angleOfView(XY, arr.shape, a=a), guess=(0.01),
                        replace_all=True, down_scale_factor=1)
