import numpy as np

class SUSD:
    """
        Implements the SUSD policy gradient method for model-free learning
        of the LQR cost
    """
    def __init__(self, model, Q, R, K0, N_rollout, T_rollout, alpha=0.1, dt=0.01, discount=1.0, term_len=100):
        self.model = model
        self.Q = Q
        self.R = R
        self.K = K0.reshape((-1,1))

        self.N_rollout = N_rollout
        self.T_rollout = T_rollout
        self.alpha = alpha
        self.discount = discount
        self.dt = dt

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

            C += (self.discount**t)*Ct

            # simulate forward in time
            x = self.model.predict(x, u)

        # return expected cost estimate and final state
        return (C, x)

    def search(self, x0, r=0.5, max_iter=5000):
        """
            Searches for the appropriate policy using the SUSD
            gradient estimation method
        """

        self.buffer = 0
        self.zmin = 0

        # store the initial policy cost
        z0, _ = self.simulate(x0[:,None], self.K)
        self.z_buf.append(z0)

        # Frobenius norm sampling
        #dK = 2*(np.random.rand(self.nb_states*self.nb_actions, self.N_rollout)-0.5)
        #dK = r*dK/np.sqrt(np.sum(abs(dK)**2,0))
        #K = self.K + dK

        # uniformly random sampling
        dK = r*2*(np.random.rand(self.nb_states*self.nb_actions, self.N_rollout)-0.5)
        K = self.K + dK

        # iterate for a maximum number of steps
        nold = None
        for it in range(max_iter):

            # compute the cost starting x0
            xt = np.ones((self.nb_states, self.N_rollout))*x0[:,None]
            z, xf = self.simulate(xt, K)

            # store the cost of this policy iteration
            self.z_buf.append(min(z))

            # perform transformation
            zmin = min(z)
            z = 1-np.exp(-(z-zmin))
            if zmin - self.zmin == 0:
                self.buffer += 1
            else:
                self.buffer = 0
                self.zmin = zmin
      
            # check if the minimum accuracy condition is met
            if self.buffer >= self.buf_size:
                self.K = K[:, np.argmin(z)]
                return (True, it)

            # compute the n-direction
            Data = K.T
            U_mean = np.mean(K, axis=1)
            R_u = Data - U_mean
            Cov = np.matmul(R_u.T, R_u)
            w, v = np.linalg.eig(Cov)
            idx = w.argsort()[::-1]
            w = w[idx]
            v = v[:,idx]
            n = v[:,-1].real

            # check if the direction reversed
            if it > 1 and np.dot(n, nold) < 0:
                n = -n
            nold = n

            # gradient update using the SUSD dynamics
            K = K + self.alpha*np.outer(n,z)
            #print it, zmin
            #print it, zmin; print K[:, np.argmin(z)]

            # provide an update occasionally
            if it % 100 == 0:
                print str(it)+"/"+str(max_iter), K[:, np.argmin(z)].flatten()
                pass

        # return convergence was not achieved
        self.K = K[:, np.argmin(z)]
        print "Warning, SUSD did not converge!"
        return (False, it)

