import numpy as np

class WB_Model:
    def __init__(self, dt):
        self.dt = dt

    def predict(self, x, u):
        # compute state update
        if len(x.shape) == 1: x = x.reshape((-1,1))
        if len(u.shape) == 1: u = u.reshape((-1,1))
        y = np.zeros_like(x)

        y[0,:] = x[0,:] + self.dt*(x[0,:] + x[1,:])
        y[1,:] = x[1,:] + self.dt*(0.01*x[0,:] + u[0,:])

        return y

    def system(self):
        A = np.array([[1, 1], [0.01, 0]])
        B = np.array([[0], [1]])

        return A, B
