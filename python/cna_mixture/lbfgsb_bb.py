import time
import numpy as np
import pylab as pl

from scipy.stats import nbinom, betabinom
from scipy.special import digamma
from scipy.optimize import approx_fprime, check_grad, minimize
from cna_mixture_rs.core import betabinom_logpmf as betabinom_logpmf_rs

RUST_BACKEND = True

np.random.seed(1234)


def pt2ab(p, t):
    return p * t, (1.0 - p) * t

def ab2pt(a, b):
    t = a + b
    p = a / t
    
    return p, t
    
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


def ln_bb_ab(ab, k, n):
    a, b = ab
    return betabinom.logpmf(k, n, a, b)


def ln_bb_pt(pt, k, n):
    a, b = pt2ab(*pt)
    return ln_bb_ab((a, b), k, n)


def grad_ln_bb_ab(ab, k, n):
    a, b = ab

    gka = digamma(k + a)
    gab = digamma(a + b)
    gnab = digamma(n + a + b)
    ga = digamma(a)

    gnkb = digamma(n - k + b)
    gb = digamma(b)

    return np.array([gka + gab - gnab - ga, gnkb + gab - gnab - gb])


def grad_ln_bb_pt(pt, k, n):
    p, t = pt
    a, b = pt2ab(p, t)

    interim = grad_ln_bb_ab((a, b), k, n)

    gradp = t * (interim[0] - interim[1])
    gradt = p * interim[0] + (1.0 - p) * interim[1]

    return np.array([gradp, gradt])


def nloglikes(x, ks, ns):
    alpha, beta = x
    return -betabinom_logpmf(ks, ns, alpha, beta)[:, 0]


def nloglike(x, ks, ns):
    return nloglikes(x, ks, ns)


if __name__ == "__main__":
    nsample = 10_000
    alphas, betas = np.array([0.6]), np.array([0.4])

    ns = np.random.randint(low=25, high=500, size=nsample)
    ks = np.array([betabinom.rvs(n, alphas[0], betas[0]) for n in ns])

    x0 = (0.9, 0.1)
    p0 = ab2pt(*x0)
    
    res = nloglike(x0, ks, ns)
    grad = grad_ln_bb_ab(x0, ks[0], ns[0])

    approx_grad = approx_fprime(
        x0, ln_bb_ab, np.sqrt(np.finfo(float).eps), ks[0], ns[0]
    )

    print(grad)
    print(approx_grad)

    grad = grad_ln_bb_pt(p0, ks[0], ns[0])
    approx_grad = approx_fprime(
	p0, ln_bb_pt, np.sqrt(np.finfo(float).eps), ks[0], ns[0]
    )

    print(grad)
    print(approx_grad)
    
    """
    epsilon = np.sqrt(np.finfo(float).eps)
    bounds = [(epsilon, None), (epsilon, None)]

    ## >>>>  L-BFGS-B no analytic gradients.
    start = time.time()
    res = minimize(
        nloglike,
        x0,
        args=(ks, ns),
        method="L-BFGS-B",
        jac=None,
        hess=None,
        hessp=None,
        bounds=bounds,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
    )
    print(
        f"\n\nOptimized with L-BFGS-B (no analytic gradients) in {time.time() - start:.3f} seconds with result:\n{res}"
    )
    """
    print("\n\nDone.\n\n")
