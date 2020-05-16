import numpy as np
import matplotlib.pyplot as plt
from NN.modeler import BB_Model
from dynamics.linear_2D import WB_Model

# define the sampling time
dt = 0.01

# define the initial state
x0 = np.array([-7, -3])

# try to load the model
nb_states = 2
nb_actions = 2
bb_model = BB_Model(nb_states, nb_actions)
loaded = bb_model.load_model(path='linear_2D_')
if loaded: print("2 Input Linear model loaded!")

wb_model = WB_Model(dt)

# define the simulation length
Nsteps = 2000

# simulate a zero-input trajectory
x_wb = np.zeros([nb_states, Nsteps])
x_bb = np.zeros([nb_states, Nsteps])
x_wb[:,0] = x0.flatten()
x_bb[:,0] = x0.flatten()
for t in range(Nsteps-1):
    u = np.zeros(2)
    #u = 20*(np.random.rand(2)-0.5)
    x_wb[:,t+1] = wb_model.predict(x_wb[:,t].reshape(-1, 1), u.reshape(-1, 1)).flatten()
    x_bb[:,t+1] = bb_model.predict(x_bb[:,t].reshape(-1, 1), u.reshape(-1, 1)).flatten()

# plot the trajectories
plt.figure()
t = np.array(range(Nsteps))*dt
leg = []
for s in range(nb_states):
    plt.plot(t, x_wb[s,:], '-')
    leg.append("$x"+str(s)+"^{(wb)}$")

for s in range(nb_states):
    plt.plot(t, x_bb[s,:], '--')
    leg.append("$x"+str(s)+"^{(bb)}$")

plt.grid()
plt.legend(leg, ncol=2)
plt.show()
