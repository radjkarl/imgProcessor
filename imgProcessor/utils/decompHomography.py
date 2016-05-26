import math



def decompHomography(H):
    '''
    @param H Homography matrix
    @returns ((translationx, translationy), rotation, (scalex, scaley), shear)
    '''
    a = H[0,0]
    b = H[0,1]
    c = H[0,2]
    d = H[1,0]
    e = H[1,1]
    f = H[1,2]
    
    p = math.sqrt(a*a + b*b)
    r = (a*e - b*d)/(p)
    q = (a*d+b*e)/(a*e - b*d)
    
    translation = (c,f)
    scale = (p,r)
    shear = q
    theta = math.atan2(b,a)
    
    return (translation, theta, scale, shear)