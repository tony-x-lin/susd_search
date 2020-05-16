import control
import numpy as np
import matplotlib.pyplot as plt
from dynamics.silvia_regulation import WB_Model
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
    #plt.plot(z_lqr, 'k--')

    plt.legend(["$NPG$ $Cost$", "$SUSD$ $Cost$", "$LQR$ $Cost$"], fontsize='medium')
    plt.title("$N_{trajectories}="+str(N_rollout)+"$, $N_{agents}="+str(N_agents)+"$, $N_{steps}="+str(N_iterations)+"$")
    plt.xlabel("$Iteration$", fontsize=10)
    plt.ylabel("$Min$ $Cost$", fontsize=10)
    #plt.xlim([0, 100])

# define sampling time
dt = 1

# define starting state
x0 = 10*np.array([1, 2])

# load model predictor
nb_states = 2
nb_actions = 2
WB_model = WB_Model()

# create the SUSD learner
Q = 20*np.eye(2)
R = 0.2*np.eye(2)
K0 = np.array([[-2., -2.], [-2., -2.]])
N_trajectories = 100
N_agents = 10
T_rollout = 200

S = SUSD(WB_model, Q, R, K0.copy(), N_agents, T_rollout, dt=dt, alpha=0.1, term_len=300)
PG = Policy_Gradient(WB_model, Q, R, K0.copy(), N_trajectories, T_rollout, dt=dt, alpha=0.8)

C_lqr = 0

# compute the optimal K
A, B = WB_model.system()
K_lqr, _, _ = control.lqr(A, B, Q, R)

# get optimal K cost
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

print "LQR Gains:"; print -K_lqr

# estiamte optimal K using SUSD
print "Search using SUSD..."
conveged, S_iters = S.search(x0, r=0.1, max_iter=10000)
K_S = S.K.reshape((nb_actions, nb_states))

# estimate optimal K using PG
print "Search using Natural Policy Gradient..."
converged, PG_iters = PG.search(x0, r=1, epsilon=0.1, max_iter=400)
K_PG = PG.K.reshape((nb_actions, nb_states))

print "Natural Policy Gradient Search took", PG_iters, "iterations"
print "Natural Policy Gradient Gains:"; print K_PG
print "SUSD Search took", S_iters, "iterations"
print "SUSD Gains:"; print K_S

# plot the cost results
cost_plot(PG.z_buf, S.z_buf, N_trajectories, N_agents, int(T_rollout/dt), C_lqr)
plt.show()
