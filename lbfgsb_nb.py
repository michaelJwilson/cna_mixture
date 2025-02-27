import numpy as np

from scipy.stats import nbinom, betabinom
from scipy.special import digamma
from scipy.optimize import approx_fprime, check_grad


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
    return r / p - k / (1.0 - p)


def grad_ln_nb_rp(x, k):
    r, p = x
    return np.array([grad_ln_nb_r(k, r, p), grad_ln_nb_p(k, r, p)])


def grad_ln_nb_mu(k, mu, var):
    p = mu / var
    r = mu * mu / (var - mu)
    
    return (p / (1.0 - p)) * grad_ln_nb_r(k, r, p) - (p * p / r) * grad_ln_nb_p(k, r, p)


def grad_ln_nb_var(k, mu, var):
    p = mu / var
    r = mu * mu / (var - mu)

    result = (p * p / (1.0 - p)) * grad_ln_nb_r(k, r, p)
    result -= (p * p * p / r / (2.0 - p)) * grad_ln_nb_p(k, r, p)

    return result


def grad_ln_nb_muvar(x, k):
    mu, var = x
    return np.array([grad_ln_nb_mu(k, mu, var), grad_ln_nb_var(k, mu, var)])


if __name__ == "__main__":
    # NB https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html
    k, r, p = 10, 12, 0.25
    x0 = np.array([r, p])

    res = ln_nb_rp(x0, k)
    grad = grad_ln_nb_rp(x0, k)
    approx_grad = approx_fprime(x0, ln_nb_rp, np.sqrt(np.finfo(float).eps), k)

    err = check_grad(ln_nb_rp, grad_ln_nb_rp, x0, k)

    assert err < 2.0e-6, f""

    mu = r * (1.0 - p) / p
    var = mu / p

    x0 = np.array([mu, var])

    grad = grad_ln_nb_muvar(x0, k)
    approx_grad = approx_fprime(x0, ln_nb_muvar, np.sqrt(np.finfo(float).eps), k)

    print(grad)
    print(approx_grad)
