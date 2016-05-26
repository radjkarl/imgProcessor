if __name__ == '__main__':
    #find out whether its faster 
    #to rather execute a function on the local GPU 
    
    from numba import guvectorize, float64,jit
    import numpy as np
    
    @guvectorize( [ (float64[:],float64[:],float64[:]) ], '(n),()->(n)',
                  nopython=True)
    def GPUfn(x,y,res):
        for i in range(x.shape[0]):
            res[i]=(x[i]**0.5+y[0]**0.5)**2
       
    @jit( [ (float64[:],float64,float64[:]) ],nopython=True)
    def CPUfn(x,y,res):
        for i in range(x.shape[0]):
            res[i]=(x[i]**0.5+y**0.5)**2  
       
    
    
         
    a= np.arange(1e7, dtype=np.float64)
    b=.010
    c=np.empty(1e7,dtype=np.float64)
    
    
    from time import  time
    t0 = time()
    for _ in xrange(10):
        GPUfn(a,b,c)
    
    t1 = time()
    for _ in xrange(10):
        CPUfn(a,b,c)
    t2 = time()
    
    
    print ('GPU', t1-t0)
    print ('CPU', t2-t1)
