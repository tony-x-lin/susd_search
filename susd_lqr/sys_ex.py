import numpy as np

A = np.array([[-2.5, -0.2, 4.3, 0.1], 
              [0.97, -10.3, 0.4, -6.1],
              [-1.2, 1.1, -4.9, 0.3],
              [0.1, 0.9, -3.4, -0.7]])

print np.linalg.eigvals(A)
