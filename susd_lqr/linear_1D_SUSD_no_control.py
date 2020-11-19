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

def cost_plot(z, n):
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

cost_plot(z_list, n)
plt.show()
