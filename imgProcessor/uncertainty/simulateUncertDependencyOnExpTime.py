import numpy as np
from scipy.signal import gaussian
from scipy.optimize.minpack import curve_fit
from fancytools.math.findXAt import findXAt

# this code was used to determine 
# PARAMS in [adjustUncertToExposureTime]
# for information can be found at...
#     ----
#     K.Bedrich: Quantitative Electroluminescence Imaging, PhD Thesis, 2017
#     Subsection 5.1.4.3: Exposure Time Dependency
#     ----


def errorDist(scale, measExpTime, n_events_in_expTime,
              event_duration, std,
              points_per_time=100, n_repetitions=300):
    '''
    TODO
    '''
    ntimes = 10
    s1 = measExpTime * scale * 10
    # exp. time factor 1/16-->16:
    p2 = np.logspace(-4, 4, 18, base=2)

    t = np.linspace(0, s1, ntimes * points_per_time * s1)

    err = None
    for rr in range(n_repetitions):

        f = _flux(t, n_events_in_expTime, event_duration, std)

        e = np.array([_capture(f, t, measExpTime, pp) for pp in p2])
        if err is None:
            err = e
        else:
            err += e
    err /= (rr + 1)
    
    # normalize, so that error==1 at 1:
    try:
        fac = findXAt(err, p2, 1)
    except:
        fac = 1

    err /= fac
    return p2, err, t, f


def exampleSignals(std=1, dur1=1, dur2=3, dur3=0.2,
                          n1=0.5, n2=0.5, n3=2):
    '''
    std ... standard deviation of every signal
    dur1...dur3 --> event duration per second
    n1...n3 --> number of events per second
    '''
    np.random.seed(123)
    t = np.linspace(0, 10, 100)

    f0 = _flux(t, n1, dur1, std, offs=0)
    f1 = _flux(t, n2, dur2, std, offs=0)
    f2 = _flux(t, n3, dur3, std, offs=0)
    return t,f0,f1,f2


def _flux(t, n, duration, std, offs=1):
    '''
    returns Gaussian shaped signal fluctuations [events]
    
    t --> times to calculate event for
    n --> numbers of events per sec
    duration --> event duration per sec
    std --> std of event if averaged over time
    offs --> event offset
    '''
    duration *= len(t) / t[-1]
    duration = int(max(duration, 1))

    pos = []
    n *= t[-1]
    pp = np.arange(len(t))
    valid = np.ones_like(t, dtype=bool)
    for _ in range(int(round(n))):
        try:
            ppp = np.random.choice(pp[valid], 1)[0]
            pos.append(ppp)
            valid[max(0, ppp - duration - 1):ppp + duration + 1] = False
        except ValueError:
            break
    sign = np.random.randint(0, 2, len(pos))
    sign[sign == 0] = -1

    out = np.zeros_like(t)

    amps = np.random.normal(loc=0, scale=1, size=len(pos))

    if duration > 2:
        evt = gaussian(duration, duration)
        evt -= evt[0]
    else:
        evt = np.ones(shape=duration)


    for s, p, a in zip(sign, pos, amps):
        pp = duration
        if p + duration > len(out):
            pp = len(out) - p

        out[p:p + pp] = s * a * evt[:pp]

    out /= out.std() / std
    out += offs
    return out


def _capture(f, t, t0, factor):
    '''
    capture signal and return its standard deviation
    #TODO: more detail
    '''
    n_per_sec = len(t) / t[-1]

    # len of one split:
    n = int(t0 * factor * n_per_sec)
    s = len(f) // n
    m = s * n
    f = f[:m]
    ff = np.split(f, s)
    m = np.mean(ff, axis=1)

    return np.std(m)


def _fitfn(x, a, b, start, end):
    '''
    decay function used to fit uncertainty change over time
    '''
    return (start - end) * (np.exp(-a * x**b)) + end



if __name__ == '__main__':
    import sys
    import pylab as plt
    
    plot = 'no_window' not in sys.argv

    t,f0,f1,f2 = exampleSignals()
    
    if plot:
        plt.figure('3 example signals with std=1')
        plt.plot(t, f0)
        plt.plot(t, f1)
        plt.plot(t, f2)

    # exposure time [sec]:
    expt = 1
    # events per sec:
    nevt = 0.05
    # event duration:
    tevt = 2
    # event std:
    std = 1
    # signallength:
    ttt = np.logspace(-4, 4, 9, base=2)
    params = []
    
    if plot:
        plt.figure('Std.Dev. scaling factors for different event durations')

    valid = np.ones_like(ttt, dtype=bool)
    
    
    for nn, tt in enumerate(ttt):
        print(tt)
        try:
            p2, err, t, f = errorDist(10, expt, nevt, tt, std)
        except:
            valid[nn] = 0
            continue
        p, perr = curve_fit(_fitfn, p2, err, 
                            (3, 1, 1, 0), 
                            max_nfev=20000,
                            bounds=((0, 0, 0, 0), 
                                    (10, 2, 10, 1)) )
        params.append(p)
        
        if plot:
            plt.plot(p2, err, label=tt)
        
    ttt = ttt[valid]
    params = np.array(params)
    
    print('parameters used by [adjustUncertToExposureTime]:')
    print(params)
    print('time factors used by [adjustUncertToExposureTime]:')
    print(ttt)

    if plot:
        plt.xlabel('Exposure time ratio (t/t0)')
        plt.gca().set_xscale("log")
        plt.gca().set_xticks(p2)
        plt.legend()

        plt.figure('Fit parameters')
        plt.plot(ttt, params[:, 0], label='a')
        plt.plot(ttt, params[:, 1], label='b')
        plt.plot(ttt, params[:, 2], label='f_o')
        plt.plot(ttt, params[:, 3], label='f_inf')
        plt.show()
