import numpy as np



def linearBlend(img1, img2, overlap, backgroundColor=None):
    '''
    Stitch 2 images vertically together.
    Smooth the overlap area of both images with a linear fade from img1 to img2

    @param img1: numpy.2dArray
    @param img2: numpy.2dArray of the same shape[1,2] as img1
    @param overlap: number of pixels both images overlap
    @returns: stitched-image
    '''
    (sizex,sizey) = img1.shape[:2]

    overlapping = True
    if overlap < 0:
        overlapping = False
        overlap = -overlap
        
    #linear transparency change:
    alpha = np.tile(np.expand_dims(np.linspace(1,0,overlap),1),sizey)
    if len(img2.shape) == 3:#multi channel img like rgb
        #make alpha 3d with n channels
        alpha = np.dstack(([alpha for _ in range(img2.shape[2])]))
    if overlapping:
        img1_cut = img1[sizex-overlap:sizex, :]
        img2_cut = img2[0:overlap,:]
    else:
        #take average of last 5 rows:
        img1_cut = np.tile( img1[ -min(sizex,5):, : ].mean(
                                axis=0),(overlap,1)).reshape(alpha.shape)
        img2_cut = np.tile(img2[ :min(img2.shape[0],5), : ].mean(
                                axis=0),(overlap,1)).reshape(alpha.shape)

    #fill intermediate area as mixture of both images  
    #################bg transparent############    
    inter = (img1_cut*alpha + img2_cut*(1-alpha)).astype(img1.dtype)
    #set background areas to value of respective other img:
    if backgroundColor != None:
        mask = np.logical_and(img1_cut==backgroundColor,
                              img2_cut!=backgroundColor)
        inter[mask] = img2_cut[mask]
        mask = np.logical_and(img2_cut==backgroundColor,
                              img1_cut!=backgroundColor)
        
        inter[mask] = img1_cut[mask]
    if not overlapping:
        overlap = 0
    return np.vstack((img1[0:sizex-overlap, :],
                      inter,
                      img2[overlap:, :]))



if __name__ == '__main__':
    from fancytools.os.PathStr import PathStr
    from imgIO import imread
    import cv2
    d = PathStr('').join('media','electroluminescence')
    i1 = d.join('EL_module_a_dist.PNG')
    i2 = d.join('EL_module_b_dist.PNG')
    
    img1 = imread(i1)/2
    img2 = imread(i2)

    cv2.namedWindow("stitched", cv2.cv.CV_WINDOW_NORMAL)
    s0 = img1.shape[0]
    for i in range(-s0,s0):
        img = linearBlend(img1, img2, i)
        cv2.imshow('stitched', img)
        cv2.waitKey(delay=20)
    cv2.waitKey()
