import time
# import torch
import numpy as np
import pylab as pl
# import torch.nn.functional as F

from scipy.stats import nbinom, betabinom
from scipy.special import digamma
from scipy.optimize import approx_fprime, check_grad, minimize
from scipy.differentiate import derivative
from cna_mixture_rs.core import nb

RUST_BACKEND = False

"""
def negative_binomial_log_pmf(k, r, p):
    log_pmf = (
        torch.lgamma(k + r)
        - torch.lgamma(k + 1)
        - torch.lgamma(r)
        + r * torch.log(p)
        + k * torch.log(1 - p)
    )
    return log_pmf
"""

def nbinom_logpmf(ks, r, p):
    if RUST_BACKEND:
        raise NotImplementedError()
    else:
        ks = np.atleast_1d(ks)        
        result = np.zeros(len(ks))

        for ii, k in enumerate(ks):
            result[ii] = nbinom.logpmf(k, r, p)

    return result
                

def ln_nb_rp(x, k):
    r, p = x
    return nbinom_logpmf(k, r, p)[0]


def muvar2rp(mu, var):
    p = mu / var
    r = mu * mu / (var - mu)

    return r, p

def rp2muvar(r, p):
    mu = r * (1. - p) / p
    var = mu / p

    return mu, var

def ln_nb_muvar(x, k):
    r, p = muvar2rp(*x)
    return nbinom.logpmf(k, r, p)


def grad_ln_nb_r(k, r, p):
    return digamma(k + r) - digamma(r) + np.log(p)


def grad_ln_nb_p(k, r, p):
    return (r / p) - k / (1.0 - p)


def grad_ln_nb_rp(x, k):
    r, p = x
    return np.array([grad_ln_nb_r(k, r, p), grad_ln_nb_p(k, r, p)])


def grad_ln_nb_mu(k, mu, var):
    r, p = muvar2rp(mu, var)

    return (2.0 * var - mu) * mu * grad_ln_nb_r(k, r, p) / (var - mu) ** 2.0 + (
        1.0 / var
    ) * grad_ln_nb_p(k, r, p)


def grad_ln_nb_var(k, mu, var):
    r, p = muvar2rp(mu, var)

    result = -(mu**2.0) * grad_ln_nb_r(k, r, p) / (var - mu) ** 2.0
    result -= (mu / var**2.0) * grad_ln_nb_p(k, r, p)

    return result


def grad_ln_nb_muvar(x, k):
    mu, var = x
    return np.array([grad_ln_nb_mu(k, mu, var), grad_ln_nb_var(k, mu, var)])


def nloglikes(r, p, samples):
    return np.array([-nbinom.logpmf(k, r, p) for k in samples])


def nloglike(x, samples):
    r, p = muvar2rp(*x)
    return nloglikes(r, p, samples).sum()


def grad_nloglike(x, samples):
    result = np.zeros(2)

    for k in samples:
        result += grad_ln_nb_muvar(x, k)

    return -result


if __name__ == "__main__":
    # NB https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html
    k, r, p = 10, 12, 0.25
    x0 = np.array([r, p])

    exp = ln_nb_rp(x0, k)
    grad = grad_ln_nb_rp(x0, k)
    approx_grad = approx_fprime(x0, ln_nb_rp, np.sqrt(np.finfo(float).eps), k)

    err = check_grad(ln_nb_rp, grad_ln_nb_rp, x0, k)
    
    assert err < 2.0e-6, f""

    mu = r * (1.0 - p) / p
    var = mu / p

    x0 = np.array([mu, var])

    res = ln_nb_muvar(x0, k)
    grad = grad_ln_nb_muvar(x0, k)
    approx_grad = approx_fprime(x0, ln_nb_muvar, np.sqrt(np.finfo(float).eps), k)

    err = check_grad(ln_nb_muvar, grad_ln_nb_muvar, x0, k)

    assert res == exp
    assert err < 2.0e-6, f""

    samples = nbinom.rvs(r, p, size=10_000)
    exp_probs = nloglikes(r, p, samples)

    # pl.plot(samples, probs, lw=0.0, marker='.')
    # pl.show()

    # NB (mu, var) = (36.0 144.0)
    x0 = np.array([50.0, 100.0])
    grad = grad_nloglike(x0, samples)

    approx_grad = approx_fprime(x0, nloglike, np.sqrt(np.finfo(float).eps), samples)
    err = check_grad(nloglike, grad_nloglike, x0, samples)

    # print(grad)
    # print(approx_grad)

    # NB err is ~ 1.0e-2
    # assert err < 2.0e-6, f""

    epsilon = np.sqrt(np.finfo(float).eps)
    bounds = [(epsilon, None), (epsilon, None)]

    ## >>>>  L-BFGS-B gradients.                                                                                                                                                                                                                                               
    start = time.time()
    res = minimize(
        nloglike,
        x0,
        args=(samples),
        method="L-BFGS-B",
        jac=grad_nloglike,
        hess=None,
        hessp=None,
        bounds=bounds,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
    )

    print(f"\n\nOptimized with L-BFGS-B in {time.time() - start:.3f} seconds with result:\n{res}")

    # r, p = muvar2rp(*res.x)
    # probs = nloglikes(r, p, samples)
    # pl.plot(samples, exp_probs, lw=0.0, marker=".")                                                                                                                                                                                                                          
    # pl.plot(samples, probs, lw=0.0, marker='.')
    # pl.show()

    exit(0)
    
    ## >>>>  L-BFGS-B no analytic gradients.
    start = time.time()
    res = minimize(
        nloglike,
        x0,
        args=(samples),
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

    print(f"\n\nOptimized with L-BFGS-B (no analytic gradients) in {time.time() - start:.3f} seconds with result:\n{res}")
    
    ## >>>>  Powell's
    start = time.time()
    res = minimize(
        nloglike,
        x0,
	args=(samples),
	method="Powell",
        jac=None,
        hess=None,
        hessp=None,
	bounds=bounds,
        constraints=(),
	tol=None,
        callback=None,
        options=None,
    )

    print(f"\n\nOptimized with Powell's in {time.time() - start:.3f} seconds with result:\n{res}")

    ## >>>>  Nelder-Mead
    start = time.time()
    res = minimize(
	nloglike,
	x0,
        args=(samples),
        method="nelder-mead",
        jac=None,
	hess=None,
        hessp=None,
	bounds=bounds,
        constraints=(),
        tol=None,
        callback=None,
	options=None,
    )

    print(f"\n\nOptimized with Nelder-Mead in {time.time() - start:.3f} seconds with result:\n{res}")
    print("\n\nDone.\n\n")
    
    """
    k = torch.tensor(k, requires_grad=False)  # Number of successes
    mean = torch.tensor(mu, requires_grad=True)  # Mean of the distribution
    var = torch.tensor(var, requires_grad=True)  # Variance of the distribution

    r = mean**2 / (var - mean)
    p = mean / var

    log_pmf = negative_binomial_log_pmf(k, r, p)
    log_pmf.backward()

    grad_mean = mean.grad
    grad_var = var.grad

    print("Log PMF:", log_pmf.item())
    print("Gradient with respect to mean:", grad_mean.item())
    print("Gradient with respect to variance:", grad_var.item())
    """
