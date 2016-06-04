from numba import jit

@jit(nopython=True)
def filterVerticalLines(arr, min_line_length=4):
    """
    Remove vertical lines in boolean array if linelength >=min_line_length
    """
    gy = arr.shape[0]
    gx = arr.shape[1]
    mn = min_line_length-1
    for i in xrange(gy):
        for j in xrange(gx):
            if arr[i,j]:
                for d in xrange(min_line_length):
                    if not arr[i+d,j]:
                        break
                if d == mn:
                    d = 0
                    while True:
                        if not arr[i+d,j]:
                            break
                        arr[i+d,j] = 0
                        d +=1       