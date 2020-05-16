import numpy as np

def step(x, u, dt):
    """ 
    Defines said's nonlinear test dynamics here as an IO function:
    x definition:
        x[0]_dot: 4*x[1]
        x[1]_dot: -x[0]^3 - 4*x[1] + u

    u definition:
        u[0]: 

    """
    
    # compute state update
    y = np.zeros_like(x)
    y[0,:] = x[0,:] + dt*4*x[1,:]
    y[1,:] = x[1,:] + dt*(-x[0,:]**3 - 4*x[1,:] + u)

    return y
