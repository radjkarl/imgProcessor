import cv2
import numpy as np
from numba import njit

#CODE ORIGIN TAKEN FROM http://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/15_Step_11.ipynb


def navierStokes2d(u, v, p, dt, nt, rho, nu,  
                boundaryConditionUV, 
                boundardConditionP, nit=100):
    '''
    solves the Navier-Stokes equation for incompressible flow
    one a regular 2d grid
    
    u,v,p --> initial velocity(u,v) and pressure(p) maps
    
    dt --> time step
    nt --> number of time steps to caluclate
    
    rho, nu --> material constants
    
    nit --> number of iteration to solve the pressure field
    '''
    #next u, v, p maps:
    un = np.empty_like(u)
    vn = np.empty_like(v)
    pn = np.empty_like(p)
    #poisson equation ==> laplace term = b[source term]
    b = np.zeros_like(p)

    ny,nx = p.shape
    #cell size:
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    
    #next time step:
    for _ in range(nt):
        un[:] = u
        vn[:] = v
        #pressure
        _buildB(b, rho, dt, u, v, dx, dy)
        for _ in range(nit):
            _pressurePoisson(p, pn, dx, dy, b)
            boundardConditionP(p)
        #UV
        _calcUV(u, v, un, p,vn,  dt, dx, dy, rho, nu)
        boundaryConditionUV(u,v)

    return u, v, p


def shiftImage(u, v, t, img, interpolation=cv2.INTER_LANCZOS4):
    '''
    remap an image using velocity field
    '''
    ny,nx = u.shape
    sy, sx = np.mgrid[:float(ny):1,:float(nx):1]
    sx += u*t
    sy += v*t
    return cv2.remap(img.astype(np.float32), 
                    (sx).astype(np.float32),
                    (sy).astype(np.float32), interpolation)


@njit
def _buildB(b, rho, dt, u, v, dx, dy):
    #[b] -> source term for poisson equation 
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))



@njit
def _pressurePoisson(p, pn, dx, dy, b):
    pn[:] = p
    p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                      (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                      b[1:-1,1:-1])


@njit
def _calcUV(u, v, un, p,vn,  dt, dx, dy, rho, nu): 
    u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                     un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx *
                   (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                   (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                    dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / dx**2 *
                   (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                   (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))   



if __name__ == '__main__':
    import sys
    from time import time
    
    cv2.namedWindow("incompressible cavity flow", cv2.WINDOW_NORMAL )

    #material:
    rho = 1
    nu = .1
    #grid:
    nx = 61
    ny = 61
    #time:
    nt = 100 #number of time steps to calculate
    dt = .001 #time step
    #initial fields:
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx)) 
    #<<<boundary conditions:
    @njit
    def boundaryConditionUV_cavity_flow(u,v):
        u[-1, :] = 10
        u[0, :] = 10
        v[:, 0] = 0
        v[:, -1] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        v[0, :] = 0
        v[-1, :]= 0

    @njit
    def boundardConditionP_cavity_flow(p):
        p[:, -1] = p[:, -2] ##dp/dy = 0 at x = 2
        p[0, :] = p[1, :]  ##dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    ##dp/dx = 0 at x = 0
        p[-1, :] = 0        ##p = 0 at y = 2
    #>>>

    #<<<create test image
    img0 = np.zeros((ny, nx))
    img0[:,:nx//2]=0.5
    img0+=0.5*np.random.rand(ny*nx).reshape(ny, nx)
    #>>>
    
    #<<<calc
    for t1 in range(1000):
        t0 = time()
        u, v, p = navierStokes2d(u, v, p, dt, nt, rho, nu, 
                              boundaryConditionUV_cavity_flow,
                              boundardConditionP_cavity_flow, nit=3)

        img0 = shiftImage(u, v, dt*nt, img0)
    #>>>

    #<<<plot
        if 'no_window'  in sys.argv:
            break
        cv2.imshow('incompressible cavity flow', img0)
        cv2.waitKey(delay=5)
        print(time()-t0)
    #>>>