import numpy as np

class WB_Model:
    def __init__(self):
        self.dt = 1
        self.A, self.B = self.system()

    def predict(self, x, u):
        # compute state update
        return np.asarray(np.matmul(self.A, x) + np.matmul(self.B, u))

    def system(self):
        A = np.array([[1.0, 0.15], 
                      [-0.2, 0.6]])

        B = np.array([[0.04, 0.01], 
                      [0.02, -0.01]])

        return A, B

    def dsystem(self):
        return self.A, self.B
