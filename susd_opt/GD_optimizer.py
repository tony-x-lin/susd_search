import numpy as np

class gd_optimizer:
    def __init__(self, f, f_grad, alpha=0.1, max_iter=5000):
        self.f = f
        self.f_grad = f_grad
        self.alpha = alpha
        self.max_iter = max_iter

    def step(self, x):
        # compute the gradient
        grad = self.f_grad(x)

        # update with gradient
        return x - self.alpha*grad, grad

    def search(self, x0, term_grad=0.001, return_hist=True):
        # setup GD search
        x = x0.copy()
        if return_hist: hist = {0: (x0, self.f(x0)[0])}

        # begin search
        for it in range(1,self.max_iter,1):
            # perform a gradient step
            x, grad = self.step(x)

            # evaluate step performance
            f = self.f(x)

            # store iteration history
            if return_hist: hist[it] = (x, f[0])

            # check termination condition
            if np.linalg.norm(grad) < term_grad:
                if return_hist: return it, hist
                else:           return it, (x, f[0])

        # search did not converge
        print "Warning, max search limit exceeded"
        if return_hist: return it, hist
        else:           return it, (x, f[0])
