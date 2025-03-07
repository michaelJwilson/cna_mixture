import numpy as np
from scipy.optimize import approx_fprime
from scipy.special import digamma
from scipy.stats import nbinom


def cost(x, k):
    mu, phi = x

    p = 1.0 / (1.0 + phi * mu)
    r = 1.0 / phi

    return nbinom.logpmf(k, r, p)


def grad_phi_cost(x, k):
    mu, phi = x

    p = 1.0 / (1.0 + phi * mu)
    r = 1.0 / phi

    return (
        (-digamma(k + r) + digamma(r)) / (phi * phi)
        + np.log(1.0 + phi * mu) / phi / phi
        + (k - phi * mu * r) / phi / (1.0 + phi * mu)
    )


def grad_mu_cost(x, k):
    mu, phi = x

    p = 1.0 / (1.0 + phi * mu)
    r = 1.0 / phi

    return k / mu - (k + r) * phi / (1.0 + phi * mu)


if __name__ == "__main__":
    x0 = (25.0, 1.0e-1)
    k = 10

    grad_phi = grad_phi_cost(x0, k)
    grad_mu = grad_mu_cost(x0, k)
    approx_grad = approx_fprime(x0, cost, np.sqrt(np.finfo(float).eps), k)

    print(grad_mu)
    print(grad_phi)
    print(approx_grad)
