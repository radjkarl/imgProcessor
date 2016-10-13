import numpy as np
import cv2


def patCircles(s0):
    '''make circle array'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    col = 255
    for rad in np.linspace(s0,s0/7.,10):
        cv2.circle(arr, (0,0), int(round(rad)), color=col, 
                   thickness=-1, lineType=cv2.LINE_AA )
        if col:
            col = 0
        else:
            col = 255
            

    return arr.astype(float)


def patCrossLines(s0):
    '''make line pattern'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    col = 255
    t = int(s0/100.)
    for pos in np.logspace(0.01,1,10):
        pos = int(round((pos-0.5)*s0/10.))
        cv2.line(arr, (0,pos), (s0,pos), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )
        cv2.line(arr, (pos,0), (pos,s0), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )

    return arr.astype(float)


def patLinCrossLines(s0):
    '''make line pattern'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    col = 255
    t = int(s0/100.)
    for pos in np.linspace(0.,s0,10):
        pos = int(round(pos))
        cv2.line(arr, (0,pos), (s0,pos), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )
        cv2.line(arr, (pos,0), (pos,s0), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )
    return arr.astype(float)



def patDiagCrossLines(s0):
    '''make line pattern'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    col = 255
    t = int(s0/50.)
    for pos in np.logspace(0.01,1,15):
        
        pos = int(round((pos-0.5)*s0/5.))
       
        cv2.line(arr,(0,pos),(pos,0), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )

        cv2.line(arr,(s0-pos,0),(s0,pos), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )
    return arr.astype(float)


def patStarLines(s0):
    '''make line pattern'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    col = 255
    t = int(s0/100.)
    for pos in np.linspace(0,np.pi/2,15):
        
        p0 = int(round(np.sin(pos)*s0*2))
        p1 = int(round(np.cos(pos)*s0*2))
       
        cv2.line(arr,(0,0),(p0,p1), color=col, 
                   thickness=t, lineType=cv2.LINE_AA )
    return arr.astype(float)


def patSiemensStar(s0, n=72, vhigh=255, vlow=0, antiasing=False):
    '''make line pattern'''
    arr = np.full((s0,s0),vlow, dtype=np.uint8)
    c = int(round(s0/2.))
    s = 2*np.pi/(2*n)
    step =  0
    for i in range(2*n): 
        p0 = round(c+np.sin(step)*2*s0)
        p1 = round(c+np.cos(step)*2*s0)
       
        step += s

        p2 = round(c+np.sin(step)*2*s0)
        p3 = round(c+np.cos(step)*2*s0)

        pts = np.array(((c,c), 
                        (p0,p1),
                        (p2,p3) ), dtype=int)

        cv2.fillConvexPoly(arr, pts,
                           color=vhigh if i%2 else vlow, 
                           lineType=cv2.LINE_AA  if antiasing else 0)
    arr[c,c]=0
    
    return arr.astype(float)


def patCycles(s0, s1=50, return_x=False):
    arr = np.zeros(s0)
    p = 1
    t = 1
    c = 0
    x,y = [],[]
    while True:
        arr[p:p+t] = 1
        p+=t
        x.append(2*t)
        y.append(p+0.5*t)
        if c > s1:
            t+=1

        c += 2
        p+=t

        if p>s0:
            break
    
    arr = arr[::-1]
    arr = np.broadcast_to(arr, (s1, s0))
    if return_x:
        #cycles/px:
        x =np.interp(np.arange(s0), y, x)  
        return arr,1/x[::-1]
    else:
        return arr
    
    
def patText(s0):
    '''make text pattern'''
    arr = np.zeros((s0,s0), dtype=np.uint8)
    s = int(round(s0/100.))
    p1 = 0
    pp1 = int(round(s0/10.))
    for pos0 in np.linspace(0,s0,10):
        cv2.putText(arr, 'helloworld', (p1,int(round(pos0))), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=s,
                    color=255, thickness=s,
                    lineType=cv2.LINE_AA )
        if p1:
            p1 = 0
        else:
            p1 = pp1
    return arr.astype(float)




if __name__ == '__main__':
    import pylab as plt 
    import sys

    res = 451

    arrs = []
    for pat in (patCircles, patCrossLines, patCycles, patDiagCrossLines,
                patLinCrossLines, patSiemensStar, patStarLines, patText):
        arrs.append(pat(res))
    
    if 'no_window' not in sys.argv: 
        for n, a in enumerate(arrs):
            plt.figure(n)
            plt.imshow(a)
    plt.show()    
    
