import numpy as np

def step(x, u, dt):
    """ 
    Defines cartpole dynamics here as an IO function:
    x definition:
        x[0]: 1D position of the cart
        x[1]: 1D velocity of the cart
        x[2]: 1D angle of the pole
        x[3]: 1D angular velocity of the pole at tip

    u definition:
        u[0]: 1D force applied to the cart

    """
    g = 9.81
    masscart = 1
    masspole = 0.1
    mass = masscart + masspole
    length = 0.5
    polemass_length = masspole*length

    # compute accelerations
    costheta = np.cos(x[2,:])
    sintheta = np.sin(x[2,:])
    temp = (u[0,:] + polemass_length*x[3,:]*x[3,:]*sintheta)/mass
    thetaacc = (g*sintheta - costheta*temp)/(length*(4.0/3.0 - masspole*costheta*costheta / mass))
    xacc = temp - polemass_length*thetaacc*costheta/mass
    
    # compute state update
    y = np.zeros_like(x)
    y[0,:] = x[0,:] + dt*x[1,:]
    y[1,:] = x[1,:] + dt*xacc
    y[2,:] = x[2,:] + dt*x[3,:]
    y[3,:] = x[3,:] + dt*thetaacc

    # clip domain of y[2]
    y[2,:] += -2*np.pi*(y[2,:] > np.pi) + 2*np.pi*(y[2,:] < -np.pi)
    return y
