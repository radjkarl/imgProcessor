import numpy as np



def extendArrayForConvolution(arr, kernelXY, 
                          modex='reflect', 
                          modey='reflect'):
    '''
    extends a given array right right border handling
    for convolution
    -->in opposite to skimage and skipy this function 
    allows to chose different mode = ('reflect', 'wrap')
    in x and y direction
    
    only supports 'warp' and 'reflect' at the moment 
    '''
    (kx, ky) = kernelXY
    kx//=2
    ky//=2
    s0,s1 = arr.shape
    
    assert ky < s0
    assert kx < s1
    
    arr2 = np.zeros((s0+2*ky, s1+2*kx), dtype=arr.dtype)
    arr2[ky:-ky,kx:-kx]=arr
    
    #original array:
    t =  arr[:ky] #TOP
    rb = arr[-1:-ky-1:-1] #reverse bottom
    rt = arr[ky-1::-1] #reverse top
    rr = arr[:,-1:-kx-1:-1] #reverse right
    l = arr[:,:kx] #left
#     rtl = arr[ky-1::-1,kx-1::-1]

    #filter array:
    tm2 = arr2[:ky ,  kx:-kx] #TOP-MIDDLE
    bm2 = arr2[-ky:,  kx:-kx]  #BOTTOM-MIDDLE
    tl2 = arr2[:ky , :kx] #TOP-LEFT
    bl2 = arr2[-ky:, :kx] #BOTTOM-LEFT
    tr2 = arr2[:ky:, -kx:]#TOP-RIGHT
    br2 = arr2[-ky:, -kx:]#TOP-RIGHT
    
    #fill edges:
    if modey == 'warp':
        tm2[:] = t
        bm2[:] = rb
  
        tl2[:] = arr2[2*ky:ky:-1,:kx]
        bl2[:] = arr2[-ky-1:-2*ky-1:-1,:kx]
    #TODO: do other options!!!  
    elif modey == 'reflect':
        tm2[:] = rt
        bm2[:] = rb
        if modex =='reflect':
            tl2[:] = arr[ky-1::-1,kx-1::-1]
            bl2[:] = arr[-1:-ky-1:-1,kx-1::-1]
            
            tr2[:] = arr[:ky,-kx:][::-1,::-1]
            br2[:] = arr[-ky:,-kx:][::-1,::-1]
            
        else:#warp
            tl2[:] = arr[ky-1::-1    , -kx:]
            bl2[:] = arr[-1:-ky-1:-1 , -kx:]
            tr2[:] = arr[ky-1::-1    , :kx]
            br2[:] = arr[-1:-ky-1:-1 , :kx]
            
    else:
        raise Exception('modey not supported')
    
    
    if modex == 'wrap':
        arr2[ky:-ky,kx-1::-1] = rr
        arr2[ky:-ky,-kx:] = l     
    elif modex == 'reflect':
        arr2[ky:-ky,:kx] = l[:,::-1]
        arr2[ky:-ky,-kx:] = rr   
    else:
        raise Exception('modex not supported')

    return arr2



if __name__ == '__main__':
    import sys
    import pylab as plt
    from imgProcessor.transform.polarTransform import linearToPolar

    s0,s1 = 150,350
    kx=101
    ky=23
    
    arr = np.fromfunction(lambda y,x: x+2*y, (s0,s1))
    
    arr2 = extendArrayForConvolution(arr, (kx,ky))

    #create an array filled with a circular function:
    c0,c1 = (24,270)
    arr3 = np.fromfunction(lambda y,x: ((x-(c1))**2+(y-(c0))**2)**0.5, (s0,s1))
    #create a polar transfrom:
    arr3lin = linearToPolar(arr3, center=(c0,c1))
    #demonstrate different border modes:
    arr4Wrong = extendArrayForConvolution(arr3lin, (kx,ky))
    arr4Right = extendArrayForConvolution(arr3lin, (kx,ky), modex='wrap')
    
    
    if 'no_window' not in sys.argv:
        
        plt.figure('normal array')
        plt.imshow(arr)
        plt.colorbar()
        plt.figure('normal array - extended')
        plt.imshow(arr2)
        plt.colorbar()
    
        plt.figure('polar array - before transform')
        plt.imshow(arr3)
        plt.colorbar()
        plt.figure('polar array - after transform')
        plt.imshow(arr3lin)
        plt.colorbar()
        
        plt.figure('polar array - modex=reflect')
        plt.imshow(arr4Wrong)
        plt.colorbar()
        plt.figure('polar array - modex=wrap')
        plt.imshow(arr4Right)
        plt.colorbar()
    
        plt.show()
        
        
        