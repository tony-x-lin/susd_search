import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_said import step

def plot(x, xhat):
    """
        plot the trajectory of the linear test model
        using matplotlib
    """
    plt.clf()
    Nsteps = x.shape[1]
    nb_states = x.shape[0]
    t = np.array(range(Nsteps))*dt
    leg = []
    for s in range(nb_states):
        plt.plot(t, x[s,:], '-')
        plt.plot(t, xhat[s,:], '--')
        leg.append("$x_"+str(s)+"$")
        leg.append("$\hat{x}_"+str(s)+"$")

    plt.grid()
    plt.legend(leg, ncol=2)
    plt.show()
    
# define sampling time
dt = 0.01

# define length of trajectory
Nsteps = 100

# define starting state
x0 = np.random.rand(2,1)

# create model predictor
nb_states = 2
nb_actions = 1
model = BB_Model(nb_states, nb_actions)
model.load_model(path='linear_said_')

# simulate the trajectory
x = np.zeros([nb_states, Nsteps])
xhat = np.zeros([nb_states, Nsteps])

x[:,0] = x0.flatten()
xhat[:,0] = x0.flatten()
for t in range(Nsteps-1):
    # choose the input
    u = np.array([0])
    #u = 15*xhat[2,t]

    # apply the predicted forward step
    xhat[:,t+1] = model.predict(x[:,t].reshape(-1, 1), u.reshape(-1, 1)).flatten()

    # apply the forward step
    x[:,t+1] = step(x[:,t].reshape(-1, 1), u.reshape(-1, 1), dt).flatten()

# plot the simulated trajectory results
plot(x, xhat)
