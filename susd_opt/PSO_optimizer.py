import numpy as np

class pso_optimizer:
    def __init__(self, f, b_up, b_lo, w=1, pp=1, pg=1, max_iter=1000):
        self.f = f
        self.b_up = b_up
        self.b_lo = b_lo
        self.w = w
        self.pp = pp
        self.pg = pg
        self.max_iter = max_iter

    def step(self, x_particles):
        x = x_particles.copy()


    def search(self, x0, term_len=100, return_hist=True):
        # setup PSO search
        x = x0.copy()
        pfmin = float('Inf')
        count = 0
        if return_hist: hist = {}

        # begin search
        for it in range(self.max_iter):
            # perform a PSO sstep
            x, xmin, fmin = self.step(x,)

            # store iteration history
            if return_hist: hist[it] = (xmin, min(fmin, pfmin))
            
            # check termination condition
            stationary = (fmin == pfmin)
            count = count*stationary + stationary
            pfmin = min(fmin, pfmin)
            if count > term_len:
                if return_hist: return it, hist
                else:           return it, (xmin, fmin)

        # search did not converge
        print "Warning, max search limit exceeded"
        if return_hist: return it, hist
        else:           return it, (xmin, fmin)
