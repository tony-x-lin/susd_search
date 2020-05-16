import control
import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_2D import WB_Model
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

def cost_plot(z):
    """
        plot training improvement
    """

    plt.figure()
    plt.grid()
    plt.plot(z)
    plt.xlabel("Iteration")
    plt.ylabel("Minimum Cost")

# define sampling time
dt = 0.01

# define simulation length
Nsteps = 1000

# define starting state
x0 = np.array([-1, 3])

# load model predictor
nb_states = 2
nb_actions = 2
bb_model = BB_Model(nb_states, nb_actions)
loaded = bb_model.load_model(path='linear_2D_')
if loaded: print("2 Input Linear model loaded!")
WB_model = WB_Model(dt)

# create the SUSD learner
Q = np.eye(2)
R = np.eye(2)
K0 = np.array([[-1., -1.], [-1., -1.]])
N_agents = 10
T_rollout = 15

S = SUSD(bb_model, Q, R, K0, N_agents, T_rollout)

# compute the optimal K
A, B = WB_model.system()
K_lqr, _, _ = control.lqr(A, B, Q, R)

# estiamte optimal K using SUSD
conveged, iters = S.search(x0)
K = S.K.reshape((nb_actions, nb_states))
print "SUSD Search took", iters, "iterations"

print "Estimated K:", K
print "LQR K:", -K_lqr

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
cost_plot(S.z_buf)
plt.show()
