import numpy as np

from imgProcessor.exceptions import EnoughImages



class Iteratives(object):
    def __init__(self, max_iter=1e4, max_dev=1e-5):
        self._max_iter = max_iter
        self._max_dev = max_dev

        self._last_dev = None
        self._n = 0

        
    def checkConverence(self, arr):
        
        dev = np.mean(arr)
        print 'residuum: %s' %dev

        #STOP ITERATION?
        if self._n > self._max_iter or (self._last_dev and (
                    (self._n> 4 and dev > self._last_dev) or dev < self._max_dev) ):
            raise EnoughImages()
