from NM_optimizer import nm_optimizer
from obj_fun import quadratic as obj

import numpy as np
import matplotlib.pyplot as plt

# create Nelder-Mead optimizer with objective function
nm_opt = nm_optimizer(obj, max_iter=10000)

# find optimal x from an initial guess using NM
N = 100
xinit = 20*np.ones([100, 1])
xstep = 30*np.eye(N)
x0 = np.hstack((xinit, xinit+xstep))

# search
nm_it, nm_hist = nm_opt.search(x0, term_dev=0.01)

print "Nelder-Mead search took", nm_it, "iterations"
print "fmin:", nm_hist[nm_it][1]

# show the results
f = np.zeros(nm_it)
for it in range(nm_it):  
    f[it] = nm_hist[it][1]

plt.plot(f, 'b-')

plt.title("Quadratic Test (100 dim)")
plt.ylabel("f(x)")
plt.xlabel("Iteration")
plt.legend(["Nelder-Mead"])
plt.grid()
plt.draw()
plt.show()
