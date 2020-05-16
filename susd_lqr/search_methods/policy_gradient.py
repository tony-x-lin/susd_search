import numpy as np
import collections

class Policy_Gradient:
    """
        Implements the (natural) policy gradient method
        for model-free learning of the LQR cost
        as described in "Global Convergence of Policy
        Gradient Methods for the Linear Quadratic 
        Regulator"
    """
    def __init__(self, model, Q, R, K0, N_rollout, T_rollout, alpha=0.01, dt=0.01, discount=1.0, term_len=200):
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
        Sigma = np.zeros((x0.shape[0], x0.shape[0]))
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

            # update Sigma
            Sigma += np.matmul(x, x.T)

            # simulate forward in time
            x = self.model.predict(x, u)

        # return expected cost estimate and final state
        return (C, Sigma, x)

    def search(self, x0, r=1, epsilon=2, max_iter=2000):  
        """
            Searches for the appropriate policy using the policy
            gradient method
        """

        # store the initial policy cost
        z0, _, _ = self.simulate(x0[:,None], self.K)
        self.z_buf.append(z0)

        # iterate for a maximum number of steps
        for it in range(max_iter):
            # Frobenius norm sampling
            dK = 2*(np.random.rand(self.nb_states*self.nb_actions, self.N_rollout)-0.5)
            dK_norm = r*dK/np.sqrt(np.sum(abs(dK)**2,0))
            K = self.K + dK_norm

            # set initial conditions for rollout
            xt = np.ones((self.nb_states, self.N_rollout))*x0[:,None]
            
            # get cost associated with policy samples
            J_, S, xf = self.simulate(xt, K)

            # estimate the (natural) gradient
            grad_hat = np.matmul(np.sum(J_*dK_norm, 1).reshape((self.nb_actions, self.nb_states)).squeeze(), np.linalg.inv(S))*K.shape[0]/(self.N_rollout*(r**2))
            #print grad_hat
            # update K accordingly
            grad_update = self.alpha*grad_hat.reshape((-1,1))
            self.K -= grad_update

            # store the cost of this policy iteration
            self.z_buf.append(min(J_))

            # check if search has converged
            self.buffer.append(np.linalg.norm(grad_hat))
            mean_grad = np.mean(list(self.buffer))
            if it > self.buf_size and abs(mean_grad) < epsilon: return (True, it)
            #print it, mean_grad; print self.K

            # provide an update occasionally
            if it % 100 == 0:
                print str(it)+"/"+str(max_iter), self.K.flatten()

        # return that convergence was not achieved
        print "Warning, policy gradient did not converge!"
        return (False, it)
