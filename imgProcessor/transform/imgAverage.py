from imgProcessor.imgIO import imread



def imgAverage(images, copy=True):
    '''
    returns an image average
    
    works on many, also unloaded images
    minimises RAM usage
    '''
    i0 = images[0]
    out = imread(i0, dtype='noUint')
    
    if copy and id(i0)==id(out):
        out = out.copy()
    
    #moving average:
    c = 2
    for i in images[1:]:
        i = imread(i, dtype='noUint')
        out += (i-out) / c
        c += 1
    
    return out