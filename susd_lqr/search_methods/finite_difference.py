import numpy as np
import collections

class Finite_Difference:
    """
        Implements the Finite Difference policy gradient method
        for model-free learning of the LQR cost
    """
    def __init__(self, model, Q, R, K0, N_rollout, T_rollout, alpha=1e-7, dt=0.01, discount=1.0, term_len=100):
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
        self.buffer = collections.deque(maxlen=term_len)

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

    def search(self, x0, r=1, epsilon=1e-3, max_iter=5000):        
        """
            Searches for the appropriate policy using the finite-difference
            gradient estimation method
        """

        self.buffer.clear()

        # iterate for a maximum number of steps
        for it in range(max_iter):
            # sample some random policies 
            #dK = r*2*(np.random.rand(self.nb_states*self.nb_actions, self.N_rollout)-0.5)

            # Frobenius norm sampling
            dK = 2*(np.random.rand(self.nb_states*self.nb_actions, self.N_rollout)-0.5)
            dK = r*dK/np.sqrt(np.sum(abs(dK)**2,0))

            # concatenate the original policy        
            dK_ = np.concatenate((dK, np.zeros((self.nb_states*self.nb_actions, 1))), axis=1)      
            K = self.K + dK
            K_ = self.K + dK_

            # set initial conditions for rollout
            xt = np.ones((self.nb_states, self.N_rollout+1))*x0[:,None]

            # get cost associated with policy samples
            J_, xf = self.simulate(xt, K_)

            # estimate the gradient using the original policy as the ref
            J = J_[0:-1]            
            J_ref = J_[-1]
            dJ = J - J_ref
            grad_hat = np.matmul(np.linalg.inv(np.matmul(dK, dK.T)), np.matmul(dK, dJ))

            # update the policy
            grad_update = self.alpha*grad_hat.reshape((-1,1))
            self.K -= grad_update
            
            # store the cost of this policy iteration
            self.z_buf.append(J_[-1])

            # check if the search has converged
            self.buffer.append(grad_update)
            mean_grad = np.mean(list(self.buffer))
            if it > self.buf_size and abs(mean_grad) < epsilon: return (True, it)
            #print it, mean_grad; print self.K
        # return that convergence was not achieved
        print "Warning, policy gradient did not converge!"
        return (False, it)
