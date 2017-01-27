import numpy as np
from collections import OrderedDict


STR_TO_DTYPE = OrderedDict((('8-bit', 'u1'),
                            ('16-bit Signed', 'i2'),
                            ('16-bit Unsigned', 'u2'),
                            ('32-bit Signed', 'i4'),
                            ('32-bit Unsigned', 'u4'),
                            ('32-bit Real/floating point', 'f4'),
                            ))



def RAW(filename, width, height, dtype, littleEndian=False):
    if dtype in STR_TO_DTYPE:
        dtype = STR_TO_DTYPE[dtype]
    
    if not littleEndian:
        dtype = '>' + dtype

    s0, s1 = width, height
    arr = np.fromfile(filename, dtype=dtype, count=s0*s1)
    try:
        arr = arr.reshape(s0, s1)  # , order='F'
    except ValueError:
        # array shape doesn't match actual size
        s1 = arr.shape[0] // s0
        arr = arr.reshape(s0, s1)
    return arr

