import control
import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_12D import WB_Model
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

def cost_plot(z, C_lqr):
    """
        plot training improvement
    """

    z_lqr = C_lqr*np.ones(len(z))

    plt.figure()
    plt.grid()
    plt.plot(z, 'b-')
    plt.plot(z_lqr, 'k--')
    plt.xlabel("Iteration")
    plt.ylabel("Minimum Cost")

# define sampling time
dt = 0.1

# define simulation length
Nsteps = 50

# define starting state
x0 = 20*np.array([1, 2, -1, -0.5])

# load model predictor
nb_states = 4
nb_actions = 3
WB_model = WB_Model(dt)

# create the SUSD learner
Q = np.eye(4)
R = np.eye(3)
K0 = np.array([[-2., -2., -2., -2.], [-2., -2., -2., -2.], [-2., -2., -2., -2.]])
N_agents = 15
T_rollout = 20

S = SUSD(WB_model, Q, R, K0, N_agents, T_rollout, dt=dt, alpha=0.05, term_len=50)

# compute the optimal K
A, B = WB_model.system()
K_lqr, _, _ = control.lqr(A, B, Q, R)

# estimate optimal K using SUSD
conveged, iters = S.search(x0, r=0.1, max_iter=1000)
K = S.K.reshape((nb_actions, nb_states))
print "SUSD Search took", iters, "iterations"

#print "Estimated K:", K
#print "LQR K:", -K_lqr
print "Error K:", K-K_lqr

# get optimal K cost
C_lqr = 0
x = x0.copy()
Nstep = int(T_rollout/dt)
for t in range(Nstep):
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

# simulate the trajectory
x = np.zeros([nb_states, Nsteps])
x_lqr = np.zeros([nb_states, Nsteps])
x[:,0] = x0.flatten()
x_lqr[:,0] = x0.flatten()
for t in range(Nsteps-1):
    u = np.dot(K, x[:,t])
    x[:,t+1] = WB_model.predict(x[:,t].reshape(-1, 1), u.reshape(-1, 1)).flatten()

    u = np.dot(-K_lqr, x_lqr[:,t])
    x_lqr[:,t+1] = WB_model.predict(x_lqr[:,t].reshape(-1, 1), u.reshape(-1, 1)).flatten()

# plot the trajectory results
plot(x, x_lqr, dt)
cost_plot(S.z_buf, C_lqr)
plt.show()

