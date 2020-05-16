from SUSD_optimizer import susd_optimizer
from NM_optimizer import nm_optimizer
from GD_optimizer import gd_optimizer
from obj_fun import quadratic as obj
from obj_fun import quadratic_grad as obj_grad

import numpy as np
import matplotlib.pyplot as plt

# create SUSD optimizer with objective function
susd_opt = susd_optimizer(obj, alpha=1E-2, max_iter=50)

# create GD optimizer with objective function and gradient function
gd_opt = gd_optimizer(obj, obj_grad, alpha=9E-1, max_iter=50)

# create Nelder-Mead optimizer with objective function
nm_opt = nm_optimizer(obj, max_iter=50)


N = 100
xinit = 5*np.ones([N, 1])


# find optimal x from an initial guess using SUSD
Nagents = N
x0 = 0.01*np.random.rand(N,Nagents) + xinit

# search
susd_it, susd_hist = susd_opt.search(x0, term_len=500)

print "SUSD search took", susd_it, "iterations"
print "fmin:", susd_hist[susd_it][2]


# find optimal x from an initial guess
x0 = 0.5*np.random.rand(N, 1) + xinit

# search
gd_it, gd_hist = gd_opt.search(x0)

print "GD search took", gd_it, "iterations"
print "fmin:", gd_hist[gd_it][1]


# find optimal x from an initial guess using NM
xstep = np.tril(10*np.ones((N, N)))
x0 = np.hstack((xinit, xinit+xstep))

# search
nm_it, nm_hist = nm_opt.search(x0, term_dev=0.01)

print "Nelder-Mead search took", nm_it, "iterations"
print "fmin:", nm_hist[nm_it][1]


# show the results
f = np.zeros(susd_it)
for it in range(susd_it):  
    f[it] = susd_hist[it][2]

plt.plot(f, '-')

f = np.zeros(gd_it)
for it in range(gd_it):  
    f[it] = gd_hist[it][1]

plt.plot(f, '-')

f = np.zeros(nm_it)
for it in range(nm_it):  
    f[it] = nm_hist[it][1]

plt.plot(f, '-')


plt.title("Quadratic ("+str(N)+" dim)")
plt.ylabel("f(x)")
plt.xlabel("Iteration")
plt.legend(["SUSD ("+str(Nagents)+" agents)", "Gradient Descent", "Nelder-Mead"])
plt.grid()
plt.draw()
plt.show()
