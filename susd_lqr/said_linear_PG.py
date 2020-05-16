import control
import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_said import WB_Model
from search_methods.policy_gradient import Policy_Gradient

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
        leg.append("$x"+str(s)+"^{(NPG)}$")

    for s in range(nb_states):
        plt.plot(t, x_lqr[s,:], '--')
        leg.append("$x"+str(s)+"^{(lqr)}$")

    plt.grid()
    plt.legend(leg, ncol=2)


def cost_plot(z):
    """
        plot training improvement
    """

    plt.figure()
    plt.grid()
    plt.plot(z)


# define sampling time
dt = 0.1

# define starting state
x0 = np.array([-1, 3])

# load model predictor
nb_states = 2
nb_actions = 1
WB_model = WB_Model(dt)

# create the finite difference learner
Q = 0.1*np.eye(2)
R = 0.1*np.array([1])
K0 = np.array([-10., -10.])
N_rollout = 40
T_rollout = 50

PG = Policy_Gradient(WB_model, Q, R, K0, N_rollout, T_rollout, dt=dt, alpha=1E-1)

# compute the optimal K
A, B = WB_model.system()
K_lqr, _, _ = control.lqr(A, B, Q, R)

# estimate the optimal K
converged, iters = PG.search(x0, epsilon=0.01, max_iter=250)
K = PG.K.reshape((nb_actions, nb_states))
print "PG Search took", iters, "iterations"

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

# plot the simulated results
plot(x, x_lqr, dt)
cost_plot(PG.z_buf)
plt.show()

