import control
import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_2D import WB_Model
from search_methods.SUSD_search import SUSD
from search_methods.policy_gradient import Policy_Gradient

def cost_plot(z_PG, z_SUSD, N_rollout, N_agents, N_iterations, C_lqr):
    """
        plot training improvement
    """

    z_lqr = C_lqr*np.ones((max(len(z_PG), len(z_SUSD))))

    plt.figure()
    plt.grid()
    plt.plot(z_PG, 'b-')
    plt.plot(z_SUSD, 'r-')
    plt.plot(z_lqr, 'k--')

    plt.legend(["$NPG$ $Cost$", "$SUSD$ $Cost$", "$LQR$ $Cost$"], fontsize='large')
    plt.title("$N_{trajectories}="+str(N_rollout)+"$ $N_{agents}="+str(N_agents)+"$ $N_{steps}="+str(N_iterations)+"$")
    plt.xlabel("$Iteration$", fontsize=15)
    plt.ylabel("$Min$ $Cost$", fontsize=15)

# define sampling time
dt = 0.1

# define starting state
x0 = np.array([-7, 3])

# load model predictor
nb_states = 2
nb_actions = 2
WB_model = WB_Model(dt)
BB_model = BB_Model(nb_states, nb_actions)
loaded = BB_model.load_model(path='linear_2D_')
if loaded: print("2 Input Linear model loaded!")

# create the SUSD learner
Q = np.eye(2)
R = np.eye(2)
K0 = np.array([[-10., -10.],[-10., -10.]])
N_trajectories = 10
N_agents = 25
T_rollout = 10

S = SUSD(WB_model, Q, R, K0.copy(), N_agents, T_rollout)
PG = Policy_Gradient(WB_model, Q, R, K0.copy(), N_trajectories, T_rollout)

# compute the optimal K
A, B = WB_model.system()
K_lqr, _, _ = control.lqr(A, B, Q, R)

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


# estimate optimal K using PG
print "Search using Natural Policy Gradient..."
converged, PG_iters = PG.search(x0, r=1)
K_PG = PG.K.reshape((nb_actions, nb_states))

# estiamte optimal K using SUSD
print "Search using SUSD..."
conveged, S_iters = S.search(x0, r=1)
K_S = S.K.reshape((nb_actions, nb_states))
        
print "LQR Gains:"; print -K_lqr
print "Natural Policy Gradient Search took", PG_iters, "iterations"
print "Natural Policy Gradient Gains:"; print K_PG
print "SUSD Search took", S_iters, "iterations"
print "SUSD Gains:"; print K_S

# plot the cost results
cost_plot(PG.z_buf, S.z_buf, N_trajectories, N_agents, int(T_rollout/dt), C_lqr)
plt.show()
