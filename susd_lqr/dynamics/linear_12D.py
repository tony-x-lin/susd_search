import numpy as np
import control

class WB_Model:
    def __init__(self, dt):
        self.dt = dt
        A, B = self.system()
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0], B.shape[1]))

        sys = control.ss(A, B, C, D)
        sysd = sys.sample(dt)
        A, B, C, D = control.ssdata(sysd)
        self.A = A
        self.B = B

    def predict(self, x, u):
        # compute state update
        return np.asarray(np.matmul(self.A, x) + np.matmul(self.B, u))

    def system(self):
        A = np.array([[-2.5, 1.2, 4.3, 0.1], 
                      [0.97, -10.3, 0.4, -6.1],
                      [-9.2, 1.1, -4.9, 0.3],
                      [1.1, 0.9, -3.4, -0.9]])

        B = np.array([[1.1, 0.4, -0.2], 
                      [-3.2, 1.4, 0.0],
                      [-0.8, 0.1, 3.0],
                      [-1.1, -0.9, 5.2]])

        return A, B

    def dsystem(self):
        return self.A, self.B
