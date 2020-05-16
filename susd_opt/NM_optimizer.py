import numpy as np

class nm_optimizer:
    def __init__(self, f, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iter=5000):
        self.f = f
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho      # rho in [0, 0.5]
        self.sigma = sigma
        self.max_iter = max_iter

    def step(self, x_vert):
        x = x_vert.copy()

        # order according to vertices
        fx = self.f(x)
        idx = np.argsort(fx)
        xsorted = x[:,idx]

        # calculate centroid of first n points
        xo = np.mean(xsorted[:,0:-1])

        # reflect stage
        xr = xo + self.alpha*(xo - xsorted[:,-1])
        fr = self.f(xr)

        # check if optimal
        fsorted = fx[idx]
        fbest = fsorted[0]
        flast = fsorted[-2]
        cond1 = (fbest <= fr)
        cond2 = (fr < flast)
        
        # return updated list
        if cond1 and cond2:
            xnew = np.hstack((xsorted[:,0:-1], xr.reshape(-1,1)))
            return (xnew, self.f(xnew))
            
        # expand stage
        if not cond1 and cond2:
            xe = xo + self.gamma*(xr - xo)
            fe = self.f(xe)
            if fe < fr:
                xnew = np.hstack((xsorted[:,0:-1], xe.reshape(-1,1)))
                return (xnew, self.f(xnew))
            
            else:
                xnew = np.hstack((xsorted[:,0:-1], xr.reshape(-1,1)))
                return (xnew, self.f(xnew))
            
        # contract stage
        if cond1 and not cond2:
            xc = xo + self.rho*(xsorted[:,-1] - xo)
            fc = self.f(xc)
            if fc < fsorted[-1]:
                xnew = np.hstack((xsorted[:,0:-1], xc.reshape(-1,1)))
                return (xnew, self.f(xnew))

        # shrink
        for i in range(1,x.shape[1],1):
            xsorted[:,i] = xsorted[:,0] + self.sigma*(xsorted[:,i]-xsorted[:,0])

        return (xsorted, self.f(xsorted))

    def search(self, x0, term_dev=0.1, return_hist=True):
        # setup NM search
        x = x0.copy()
        if return_hist: hist = {}

        # begin search
        for it in range(self.max_iter):
            # perform an NM step
            x, f = self.step(x)

            # store iteration history
            if return_hist: hist[it] = (x, f[0])

            # check termination condition
            if np.std(f) < term_dev:
                if return_hist: return it, hist
                else:           return it, (x, f[0])

        # search did not converge
        print "Warning, max search limit exceeded"
        if return_hist: return it, hist
        else:           return it, (x, f[0])



