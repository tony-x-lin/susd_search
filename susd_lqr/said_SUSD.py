import numpy as np
import matplotlib.pyplot as plt
#from SUSD.SUSD_search import SUSD
from SUSD.SUSD_nn_cost_search import SUSD
from dynamics.said import step
#from NN.modeler import BB_Model
from NN.cost_modeler import BB_Cost_Model

def plot(x, u):
    """
        plot the trajectory of the system 
        using matplotlib
    """
    plt.clf()
    Nsteps = x.shape[1]
    nb_states = x.shape[0]
    t = np.array(range(Nsteps))*dt
    leg = []
    #for s in range(nb_states):
    #    plt.plot(t, x[s,:], '-')
    #    leg.append("$x_"+str(s)+"$")

    #plt.plot(t, u.flatten())
    #leg.append("$u$")
    plt.plot(x[0,:], x[1,:], '-')
    plt.plot(x[0,-1], x[1,-1], 'o')
    plt.grid()
    #plt.legend(leg, ncol=2)
    plt.show()

# define sampling time
dt = 0.01

# instantiate the SUSD learner
nb_states = 2
nb_actions = 1
nb_gains = 2
model = BB_Cost_Model(nb_states, nb_gains)
model.load_model(path='said_')

# define SUSD learner parameters
Ntrajectories = 8
Q = np.diag([1, 1])
R = np.array([1])
tf = 5
tf_learning = 5
window = 20
K0 = np.array([[1], [1]])
S = SUSD(Q, R, K0, dt, Ntrajectories, tf, window, model, accuracy=0.1)

# define experiment parameters
Nsteps = 1000
tlin = 0
linearization_range = 0.01

# define the initial condition
x0 = np.array([3, 3])
#x0 = 5*(np.random.rand(1,2)-0.5)
#x0 = np.array([0, 0])
x = np.zeros([nb_states, Nsteps])
u = np.zeros([nb_actions, Nsteps])
x[:,0] = x0.flatten()

# use the SUSD search to find the first policy 
Khat = S.search(x[:,0])
for t in range(Nsteps-1):
    # check if the search needs to be re-computed
    Khat = S.search(x[:,t])
    #if np.linalg.norm(x[:,t]-x[:,tlin]) > linearization_range:
    #    Khat = S.search(x[:,t])
    #    tlin = t

    # apply the policy and step forward
    u[:,t] = np.dot(Khat, x[:,t])

    #u[:,t] = 0    
    x[:,t+1] = step(x[:,t].reshape(-1, 1), u[:,t].reshape(-1, 1), dt).flatten()
    print t, np.array2string(x[:,t+1], precision=2)
    # visualize in real-time
    #if t % 10 == 0: visualize(x[:,t+1])

# plot the trajectory results
plot(x, u)
