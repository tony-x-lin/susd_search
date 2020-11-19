import numpy as np

class susd_optimizer:
    def __init__(self, f, alpha=0.5, k=1, max_iter=5000, sig=0):
        self.f = f
        self.alpha = alpha
        self.k = k
        self.max_iter = max_iter
        self.sig = sig

    def step(self, x, nold, idminold):
        # perform a function evaluation of the current x estimate
        fx_ = self.f(x) 
        fx = fx_ + self.sig*np.random.randn(x.shape[1])

        # perform the function transformation
        idmin, fmin = np.argmin(fx), min(fx); print x
        fx = 1-np.exp(-self.k*(fx-fmin))

        # check if the same index is still the minimum        
        stationary = (idmin == idminold)
        idminold = idmin

        # compute the n direction
        Data = x.T
        U_mean = np.mean(x, axis=1)
        R_u = Data - U_mean
        Cov = np.matmul(R_u.T, R_u)
        w, v = np.linalg.eig(Cov)
        idx = w.argsort()[::-1]
        v = v[:,idx]; n = v[:,-1].real

        # check if the direction reversed
        n = np.sign(np.dot(n, nold))*n
        nold = n

        # step using the SUSD method
        x = x + self.alpha*np.outer(n, fx)

        # return if the same minimum was found
        return stationary, x, min(fx_), nold, idminold

    def search(self, x0, term_len=200, return_hist=True):
        # setup SUSD search
        count = 0
        pfmin = float('Inf')
        nold = np.random.rand(x0.shape[0])
        idminold = -1
        x = x0.copy()
        if return_hist: hist = {}

        # begin search
        for it in range(self.max_iter):
            # perform an SUSD step
            stationary, x, fmin, nold, idminold = self.step(x, nold.copy(), idminold)

            # store iteration history
            if return_hist: hist[it] = (x, x[:,idminold], min(fmin, pfmin))
            pfmin = min(fmin, pfmin)

            # check termination condition
            if stationary: count += 1
            else: count = 0

            if count == term_len: 
                if return_hist: return it, hist
                else:           return it, (x[:,idminold], fmin)

        # search did not converge
        print "Warning, max search limit exceeded"
        if return_hist: return it, hist
        else:           return it, (x[:,idminold], fmin)


