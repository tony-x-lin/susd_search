import control
import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_1D_unstable import WB_Model
from search_methods.SUSD_search import SUSD

def plot(x, x_lqr, dt):
    """
        plot the trajectory of the linear test model
        using matplotlib
    """
    plt.figure()
    Nsteps = x.shape[1]
    nb_states = x.shape[0]
    t = np.array(range(Nsteps))*dt
    leg = []
    for s in range(nb_states):
        plt.plot(t, x[s,:], '-')
        leg.append("$x"+str(s)+"^{(susd)}$")

    for s in range(nb_states):
        plt.plot(t, x_lqr[s,:], '--')
        leg.append("$x"+str(s)+"^{(lqr)}$")

    plt.grid()
    plt.legend(leg, ncol=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")

def cost_plot(z, C_lqr, n):
    """
        plot training improvement
    """
    leg = []
    max_z = 0
    plt.figure()
    plt.grid()
    for idx in range(len(z)):
        max_z = max(max_z, len(z[idx]))
        plt.plot(np.array(z[idx])/(n[idx]/0.01), '-')
        leg.append("Minimum SUSD Cost ("+str(n[idx])+")")

    leg.append("LQR Minimum Cost")
    z_lqr = C_lqr*np.ones(max_z)
    plt.plot(z_lqr, 'k--')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Minimum Cost", fontsize=10)
    plt.ylim([1E1, 1E15])
    plt.yscale("log")    
    #plt.legend(["SUSD Minimum Cost","LQR Minimum Cost"])
    plt.legend(leg)
    plt.title("Optimization curves with varying numbers of agents")

# define sampling time
dt = 0.01

n = [2, 5, 8, 10, 15]
z_list = [0] * len(n)
for idn in range(len(n)):
    # define simulation length
    Nsteps = 500

    # define starting state
    x0 = np.array([-8, 9])

    # load model predictor
    nb_states = 2
    nb_actions = 1
    wb_model = WB_Model(dt)

    # create the SUSD learner
    Q = np.eye(2)
    R = np.array([1])
    K0 = np.array([0.1, 0.1])
    N_rollout = n[idn]
    T_rollout = 20

    S = SUSD(wb_model, Q, R, K0, N_rollout, T_rollout, alpha=0.35, term_len=100)

    # estimate optimal K using SUSD
    conveged, iters = S.search(x0, max_iter=100)
    K = S.K.reshape((nb_actions, nb_states))
    print "SUSD Search took", iters, "iterations"
    print "Estimated K:"; print K
    z_list[idn] = S.z_buf

C_lqr = 0
# compute the optimal K
A, B = wb_model.system()
K_lqr, _, _ = control.lqr(A, B, Q, R)

# compute optimal cost
x = x0.copy()
Nsteps = int(T_rollout/dt)
for t in range(Nsteps):
    # compute input
    u = np.dot(-K_lqr, x)

    # compute cost
    Ct = 0
    if nb_states == 1: Ct += (x*Q*x).flatten()
    else: Ct += np.sum(x*np.matmul(Q, x), axis=0)

    if nb_actions == 1: Ct += (u*R*u).flatten()
    else: Ct += np.sum(u*np.matmul(R, u), axis=0)

    C_lqr += Ct

    # simulate forward in time
    x = wb_model.predict(x, u).flatten()

print "LQR K:"; print -K_lqr

# simulate the trajectory
#    x = np.zeros([nb_states, Nsteps])
#    x_lqr = np.zeros([nb_states, Nsteps])
#    x[:,0] = x0.flatten()
#    x_lqr[:,0] = x0.flatten()
#    for t in range(Nsteps-1):
#        u = np.dot(K, x[:,t])
#        x[:,t+1] = wb_model.predict(x[:,t].reshape(-1, 1), u.reshape(-1, 1)).flatten()

#        u = np.dot(-K_lqr, x_lqr[:,t])
#        x_lqr[:,t+1] = wb_model.predict(x_lqr[:,t].reshape(-1, 1), u.reshape(-1, 1)).flatten()

# plot the trajectory results
#    plot(x, x_lqr, dt)
cost_plot(z_list, C_lqr/Nsteps, n)
plt.show()
