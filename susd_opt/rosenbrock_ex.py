from SUSD_optimizer import susd_optimizer
from NM_optimizer import nm_optimizer
from GD_optimizer import gd_optimizer
from obj_fun import rosenbrock as obj
from obj_fun import rosenbrock_grad as obj_grad

import numpy as np
import matplotlib.pyplot as plt

# create SUSD optimizer with objective function
#susd_opt = susd_optimizer(obj, alpha=0.5, max_iter=10000) #7E-3

# create GD optimizer with objective function and gradient function
#gd_opt = gd_optimizer(obj, obj_grad, alpha=1E-1, max_iter=50)

# create Nelder-Mead optimizer with objective function
#nm_opt = nm_optimizer(obj, max_iter=50)


sig = [0,1,2,3,4]
N = 25
xinit = 5*np.ones([N, 1])
f = []
leg = []
for ids in range(len(sig)):
    # create SUSD optimizer with objective function
    susd_opt = susd_optimizer(obj, alpha=0.8, max_iter=100, sig=sig[ids]) #7E-3

    # find optimal x from an initial guess
    Nagents = 2*N
    x0 = 0.5*np.random.rand(N,Nagents) + xinit

    # search
    susd_it, susd_hist = susd_opt.search(x0, term_len=50)

    print "SUSD search took", susd_it, "iterations"
    print "fmin:", susd_hist[susd_it][2]

    f = np.zeros(susd_it)
    for it in range(susd_it):
        f[it] = susd_hist[it][2]

    plt.plot(f, '-')
    leg.append("Sig="+str(sig[ids]))

# find optimal x from an initial guess
#x = 0.5*np.random.rand(N, 1) + xinit

# search
#gd_it, gd_hist = gd_opt.search(x, term_grad=0.1)

#print "GD search took", gd_it, "iterations"
#print "fmin:", gd_hist[gd_it][1]


# find optimal x from an initial guess using NM
#xstep = np.tril(10*np.ones((N, N)))
#x0 = np.hstack((xinit, xinit+xstep))

# search
#nm_it, nm_hist = nm_opt.search(x0, term_dev=0.01)

#print "Nelder-Mead search took", nm_it, "iterations"
#print "fmin:", nm_hist[nm_it][1]


# show the results
#f = np.zeros(susd_it)
#for it in range(susd_it):  
#    f[it] = susd_hist[it][2]

#plt.plot(f, '-')
plt.plot(np.zeros(susd_it)+100, 'k--')
#f = np.zeros(gd_it)
#for it in range(gd_it):  
#    f[it] = gd_hist[it][1]

#plt.plot(f, '-')

#f = np.zeros(nm_it)
#for it in range(nm_it):  
#    f[it] = nm_hist[it][1]

#plt.plot(f, '-')

leg.append("Rosenbrock Minimum")

plt.title("Rosenbrock ("+str(N)+" dim)")
plt.ylabel("f(x)")
plt.xlabel("Iteration")
#plt.legend(["SUSD ("+str(Nagents)+" agents)", "Rosenbrock Minimum"])
plt.legend(leg)
plt.grid()
plt.draw()
plt.show()
