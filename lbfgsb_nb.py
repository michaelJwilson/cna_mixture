import torch
import numpy as np
import pylab as pl
import torch.nn.functional as F

from scipy.stats import nbinom, betabinom
from scipy.special import digamma
from scipy.optimize import approx_fprime, check_grad, minimize
from scipy.differentiate import derivative


def negative_binomial_log_pmf(k, r, p):
    log_pmf = (
        torch.lgamma(k + r)
        - torch.lgamma(k + 1)
        - torch.lgamma(r)
        + r * torch.log(p)
        + k * torch.log(1 - p)
    )
    return log_pmf


def ln_nb_rp(x, k):
    r, p = x
    return nbinom.logpmf(k, r, p)


def ln_nb_muvar(x, k):
    mu, var = x

    p = mu / var
    r = mu * mu / (var - mu)

    return nbinom.logpmf(k, r, p)


def grad_ln_nb_r(k, r, p):
    return digamma(k + r) - digamma(r) + np.log(p)


def grad_ln_nb_p(k, r, p):
    return (r / p) - k / (1.0 - p)


def grad_ln_nb_rp(x, k):
    r, p = x
    return np.array([grad_ln_nb_r(k, r, p), grad_ln_nb_p(k, r, p)])


def grad_ln_nb_mu(k, mu, var):
    p = mu / var
    r = mu * mu / (var - mu)

    return (2.0 * var - mu) * mu * grad_ln_nb_r(k, r, p) / (var - mu) ** 2.0 + (
        1.0 / var
    ) * grad_ln_nb_p(k, r, p)


def grad_ln_nb_var(k, mu, var):
    p = mu / var
    r = mu * mu / (var - mu)

    result = -(mu**2.0) * grad_ln_nb_r(k, r, p) / (var - mu) ** 2.0
    result -= (mu / var**2.0) * grad_ln_nb_p(k, r, p)

    return result


def grad_ln_nb_muvar(x, k):
    mu, var = x
    return np.array([grad_ln_nb_mu(k, mu, var), grad_ln_nb_var(k, mu, var)])


def nloglikes(r, p, samples):
    return np.array([-nbinom.logpmf(k, r, p) for k in samples])


def nloglike(x, samples):
    mu, var = x    
    p = mu / var
    r = mu * mu / (var - mu)

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
    x0 = np.array([50., 100.])
    grad = grad_nloglike(x0, samples)

    approx_grad = approx_fprime(x0, nloglike, np.sqrt(np.finfo(float).eps), samples)
    err = check_grad(nloglike, grad_nloglike, x0, samples)
    
    print(grad)
    print(approx_grad)

    # assert err < 2.0e-6, f""
    
    epsilon = np.sqrt(np.finfo(float).eps)
    bounds = [(epsilon, None), (epsilon, None)]
    
    # NB L-BFGS-B accepts bounds.
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

    print(res)

    probs = nloglikes(*res.x, samples)

    pl.plot(samples, probs, lw=0.0, marker='.')
    pl.plot(samples, probs, lw=0.0, marker='.')
    pl.show()

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
