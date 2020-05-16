import numpy as np

"""
    objective functions must map from R^{M x N} to R^{N} where M is length of x
    and N is number of SUSD agents/samples to take
"""

def quadratic(x):
    return np.linalg.norm(x,axis=0)**2 + 100

def quadratic_grad(x):
    return 2*x

def rastrigin(x):
    A = 0.1
    M = x.shape[0]
    acos = A*np.cos(2*np.pi*x)
    xi_diff = x**2 - acos
    return 0*A*M + np.sum(xi_diff, axis=0) + 100

def rastrigin_grad(x):
    A = 0.1
    return 2*x + A*np.sin(2*np.pi*x)

def rosenbrock(x0):
    M = 0.01
    x = x0.copy()
    if len(x.shape) == 1: x = x.reshape(-1,1)
    xshift = np.roll(x, -1, axis=0)
    xclip = x[0:-1,:]
    xshiftclip = xshift[0:-1,:]
    
    xdiff = M*(xshiftclip-xclip**2)**2 + (1-xclip)**2
    return np.sum(xdiff, axis=0) + 100

def rosenbrock_grad(x):
    M = 0.01
    xdshift = np.roll(x,  1, axis=0)
    xushift = np.roll(x, -1, axis=0)
    grad_x = -4*M*(xushift-x**2)*x - 2*(1-x) + 2*M*(x-xdshift**2)

    grad_x[0,:] = -4*M*(xushift[0,:]-x[0,:]**2)*x[0,:] - 2*(1-x[0,:])
    grad_x[-1,:] = 2*M*(x[-1,:]-xdshift[-1,:]**2)
    return grad_x
