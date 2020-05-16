import control
import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_high import WB_Model
from search_methods.SUSD_search import SUSD

def plot(alpha, z, iters, C_lqr):
    """
        plot the effect of changing alpha
    """

    z_lqr = C_lqr*np.ones(len(z))

    plt.figure()
    plt.title("Cost vs Learning Rate")
    plt.grid()
    plt.plot(alpha, z, 'b-')
    plt.plot(alpha, z_lqr, 'k--')
    plt.ylim([4000, 7000])
    plt.xlabel("Alpha Parameter")
    plt.ylabel("Learned Cost")
    
    plt.figure()
    plt.title("Iterations vs Learning Rate")
    plt.grid()
    plt.plot(alpha, iters, 'b-')
    plt.xlabel("Alpha Parameter")
    plt.ylabel("Iterations")

# define sampling time
dt = 0.1

# define simulation length
Nsteps = 500

# define starting state
x0 = 20*np.array([1, 1, 1, 1])

# load model predictor
nb_states = 4
nb_actions = 2
WB_model = WB_Model(dt)

# compute the optimal K
Q = np.eye(4)
R = np.eye(2)
A, B = WB_model.system()
sys = control.StateSpace(A, B, np.eye(4), np.zeros((4,2)), 1)
K_lqr, _, _ = control.lqr(sys, Q, R)
T_rollout = 10

# get optimal K cost
C_lqr = 0
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
    x = WB_model.predict(x, u).flatten()


# create the SUSD learner
N_agents = 30
K0 = np.array([[-2., -2., -2., -2.], [-2., -2., -2., -2.]])

alpha = 0.01+np.linspace(0,0.74,100); alpha = alpha[::-1]
#alpha = [2., 1.5, 1., 0.5, 0.1, 0.05, 0.01]
Z = []
s_it = []
for it in range(alpha.shape[0]):
    S = SUSD(WB_model, Q, R, K0, N_agents, T_rollout, dt=dt, alpha=alpha[it], term_len=400)

    # estimate optimal K using SUSD
    conveged, iters = S.search(x0, r=0.1, max_iter=1000)
    s_it.append(iters)
    Z.append(S.z_buf[-1])
    print "Alpha="+str(alpha[it])+" took", iters, "iterations", "f="+str(S.z_buf[-1])

# plot the trajectory results
plot(alpha[::-1], Z, s_it, C_lqr)
plt.show()

