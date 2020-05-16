import matplotlib.pyplot as plt
import numpy as np

def cost_plot(z, names):
    """
        plot training improvement
    """

    plt.figure()
    plt.grid()
    for i in range(len(z)):
        plt.plot(z[i])

    plt.legend(names)
    plt.title("Cost during Training")

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
        leg.append("$x"+str(s)+"$")

    for s in range(nb_states):
        plt.plot(t, x_lqr[s,:], '--')
        leg.append("$x"+str(s)+"^{(lqr)}$")

    plt.grid()
    plt.legend(leg, ncol=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("System Trajectory vs lqr")
