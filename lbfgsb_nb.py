import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import nbinom, betabinom
from scipy.special import digamma
from scipy.optimize import approx_fprime, check_grad
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

    return (2. * var - mu) * mu * grad_ln_nb_r(k, r, p) / (var - mu)**2. + (1. / var) * grad_ln_nb_p(k, r, p)


def grad_ln_nb_var(k, mu, var):
    p = mu / var
    r = mu * mu / (var - mu)

    result = -mu **2. * grad_ln_nb_r(k, r, p) / (var - mu)**2.
    result -= (mu / var**2.) * grad_ln_nb_p(k, r, p)

    return result


def grad_ln_nb_muvar(x, k):
    mu, var = x
    return np.array([grad_ln_nb_mu(k, mu, var), grad_ln_nb_var(k, mu, var)])



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
