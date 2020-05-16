import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import RBF, ConstantKernel as C


def rbf_kernel(x, y, sig, l):
    '''
        Input:
            x: (m x d) array
            y: (n x d) array

        Output:
            K: (m x n) array
    '''

    sqdist = np.sum(x**2, 1).reshape(-1,1) + np.sum(y**2, 1) - 2*np.dot(x, y.T)
    return sig**2 * np.exp(-0.5/l**2 * sqdist)



class Gaussian_Process:
    def __init__(self, sigma=1, l=1):
        self.sigma = sigma
        self.l = l
        
    def 
