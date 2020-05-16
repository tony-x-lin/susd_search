import control
import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_said import WB_Model
from search_methods.SUSD_search import SUSD
from search_methods.random_search import Random_Search
from search_methods.policy_gradient import Policy_Gradient

def cost_plot(z_PG, z_SUSD, z_RS, C_lqr, N_rollout, N_agents, N_iterations):
    """
        plot training improvement
    """

    z_lqr = C_lqr*np.ones((max(len(z_PG), len(z_SUSD))))

    plt.figure()
    plt.grid()
    plt.plot(z_PG, 'b-')
    plt.plot(z_SUSD, 'r-')
    plt.plot(z_RS, 'g-')

    plt.legend(["NPG Cost", "SUSD Cost", "RS Cost"])
    plt.title("$N_{trajectories}="+str(N_rollout)+"$ $N_{agents}="+str(N_agents)+"$ $N_{steps}="+str(N_iterations)+"$")

# define sampling time
dt = 0.1

# define starting state
x0 = np.array([-7, 3])

# load model predictor
nb_states = 2
nb_actions = 1
WB_model = WB_Model(dt)

# create the SUSD learner
Q = np.eye(2)
R = np.array([1])
K0 = np.array([-10., -10.])
N_points = 10
N_trajectories = 100
N_agents = 5
T_rollout = 10

RS = Random_Search(WB_model, Q, R, K0.copy(), N_points, T_rollout, 0.01*np.eye(2), dt=dt)
S = SUSD(WB_model, Q, R, K0.copy(), N_agents, T_rollout, dt=dt, alpha=0.9)
PG = Policy_Gradient(WB_model, Q, R, K0.copy(), N_trajectories, T_rollout, dt=dt, alpha=9E-1)

# compute the optimal K
A, B = WB_model.system()
K_lqr, _, _ = control.lqr(A, B, Q, R)
print "LQR Gains:", -K_lqr

# estimate optimal K using FD
#converged, iters = FD.search(x0, r=0.5)
#K = FD.K.reshape((nb_actions, nb_states))
#print "Finite Difference Search took", iters, "iterations"
#print "Finite Difference Gains:", K

# estimate optimal K using RS
converged, iters = RS.search(x0.copy(), max_iter=250)
K = RS.K.reshape((nb_actions, nb_states))
print "Random Search took", iters, "iterations"
print "Random Search Gains:", K

# estimate optimal K using PG
converged, iters = PG.search(x0.copy(), r=0.5, epsilon=0.1, max_iter=250)
K = PG.K.reshape((nb_actions, nb_states))
print "Natural Policy Gradient Search took", iters, "iterations"
print "Natural Policy Gradient Gains:", K

# estiamte optimal K using SUSD
conveged, iters = S.search(x0.copy(), r=0.5, max_iter=250)
K = S.K.reshape((nb_actions, nb_states))
print "SUSD Search took", iters, "iterations"
print "SUSD Gains:", K

C_lqr = 0
x = x0.copy()
Nsteps = int(T_rollout/dt)
for t in range(Nsteps-1):
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

# plot the trajectory results
cost_plot(PG.z_buf, S.z_buf, RS.z_buf, C_lqr, N_trajectories, N_agents, int(T_rollout/dt))
plt.show()
