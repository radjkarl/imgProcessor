
#TODO: embed for FitHistogramPeaks
def poisson(x, a, b, c, d=0):
    '''
    Poisson function
    a -> height of the curve's peak
    b -> position of the center of the peak
    c ->  standard deviation 
    d -> offset
    '''
    from scipy.misc import factorial
    lamb = 1
    X = (x/(2*c)).astype(int)
    return a * (( lamb**X/factorial(X)) * np.exp(-lamb) ) +d
