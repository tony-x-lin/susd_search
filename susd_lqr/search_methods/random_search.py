import numpy as np

class Random_Search:
    """
        Implements the random search method for model-free learning
        of the LQR cost
    """
    def __init__(self, model, Q, R, K0, N_points, T_rollout, cov, dt=0.01, term_len=100):
        self.model = model
        self.Q = Q
        self.R = R
        self.K = K0.reshape((-1,1))

        self.dt = dt
        self.T_rollout = T_rollout
        self.N_points = N_points
        self.cov = cov

        self.nb_states = Q.shape[0]
        self.nb_actions = R.shape[0]

        # circular buffer for termination condition
        self.buf_size = term_len
        self.buffer = 0
        self.zmin = 0

        # buffer for cost during search
        self.z_buf = []

        # helper matrix
        self.O = np.zeros((self.nb_actions, self.nb_actions*self.nb_states))
        for i in range(self.nb_actions):
            for j in range(self.nb_actions):
                self.O[i, j*self.nb_states:(j+1)*self.nb_states] = (i==j)*np.ones(self.nb_states)

    def simulate(self, x0, K0):
        """
            Performs a simulation for a batch of agents using the provided
            black-box model.
            x0 is self.nb_states x N
            K0 is self.nb_states*self.nb_actions x N
        """

        # prepare simulation
        Nsteps = int(self.T_rollout/self.dt)
        C = np.zeros(x0.shape[1])
        x = x0.copy()

        # get batch simulation
        for t in range(Nsteps):
            # compute batch input
            xtmp = np.tile(x, (self.nb_actions,1))
            u = np.matmul(self.O, K0*xtmp)

            # compute cost
            Ct = np.zeros(x0.shape[1])
            if self.nb_states == 1: Ct += (x*self.Q*x).flatten()
            else: Ct += np.sum(x*np.matmul(self.Q, x), axis=0)

            if self.nb_actions == 1: Ct += (u*self.R*u).flatten()
            else: Ct += np.sum(u*np.matmul(self.R, u), axis=0)

            C += Ct

            # simulate forward in time
            x = self.model.predict(x, u)

        # return expected cost estimate and final state
        return (C, x)

    def search(self, x0, max_iter=5000):
        """
            Searches for the appropriate policy using the random search method
        """

        self.buffer = 0
        self.zmin = float('Inf')
        K_best = self.K

        # iterate for a maximum number of steps
        for it in range(max_iter):
            # gaussian random sampling
            dK = np.random.multivariate_normal(0*K_best.flatten(), self.cov, self.N_points).T
            K = K_best + dK

            # compute the cost starting x0
            xt = np.ones((self.nb_states, self.N_points))*x0[:,None]
            z, xf = self.simulate(xt, K)

            # store information
            if min(z) >= self.zmin:
                self.buffer += 1
            else:
                self.buffer = 0
                idK = np.argmin(z)
                K_best = K[:, np.argmin(z)].reshape((-1,1))
                self.zmin = min(z)

            # store the cost of this policy iteration
            self.z_buf.append(self.zmin)
     
            # check if the minimum accuracy condition is met
            if self.buffer >= self.buf_size:
                self.K = K_best
                return (True, it)

            # provide an update occasionally
            if it % 100 == 0:
                print str(it)+"/"+str(max_iter), K[:, np.argmin(z)].flatten()
                pass

        # return convergence was not achieved
        self.K = K_best
        print "Warning, random search did not converge!"
        return (False, it)

