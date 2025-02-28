import time
import numpy as np
import pylab as pl

from scipy.stats import nbinom, betabinom
from scipy.special import digamma
from scipy.optimize import approx_fprime, check_grad, minimize
from cna_mixture_rs.core import betabinom_logpmf as betabinom_logpmf_rs

RUST_BACKEND = False

np.random.seed(1234)

def betabinom_logpmf(ks, ns, alphas, betas):
    ks = np.atleast_1d(ks).astype(float)
    ns = np.atleast_1d(ns).astype(float)
    
    alphas = np.atleast_1d(alphas).astype(float)
    betas = np.atleast_1d(betas).astype(float)
    
    if RUST_BACKEND:
        ks = np.ascontiguousarray(ks)
        ns = np.ascontiguousarray(ns)
        
        alphas = np.ascontiguousarray(alphas)
        betas = np.ascontiguousarray(betas)
        
        result = betabinom_logpmf_rs(ks, ns, betas, alphas)
        result = np.array(result)
        
    else:
        result = np.zeros(shape=(len(ks), len(alphas)))

        for ss, (a, b) in enumerate(zip(alphas, betas)):
            for ii, (k, n) in enumerate(zip(ks, ns)):
                result[ii, ss] = betabinom.logpmf(k, n, b, a)
                
    return result


if __name__ == "__main__":
    alphas, betas = np.array([0.6]), np.array([0.4])

    nsample = 10_000
    
    ns = np.random.randint(low=25, high=500, size=nsample)
    samples = np.array([[betabinom.rvs(n, alphas[0], betas[0]), n] for n in ns])
    
    ln_probs = betabinom_logpmf(samples[:,0], samples[:,1], alphas, betas)
    
    print(ln_probs) 
    print("\n\nDone.\n\n")
