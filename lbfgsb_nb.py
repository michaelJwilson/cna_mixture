import numpy as np

from scipy.stats import nbinom, betabinom
from scipy.special import digamma
from scipy.optimize import approx_fprime, check_grad


def func_ln_nb(x, k):
    r, p = x
    return nbinom.logpmf(k, r, p)

def grad_r_ln_nb(k, r, p):
    return digamma(k + r) - digamma(r) + np.log(p)

def grad_p_ln_nb(k, r, p):
    return r / p - k / (1.0 - p)


def grad_ln_nb(x, k):
    r, p = x
    return np.array([grad_r_ln_nb(k, r, p), grad_p_ln_nb(k, r, p)])


if __name__ == "__main__":
    # NB https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html
    k, r, p = 10, 12, 0.25
    x0 = np.array([r, p])

    res = func_ln_nb(x0, k)
    grad = grad_ln_nb(x0, k)
    approx_grad = approx_fprime(x0, func_ln_nb, np.sqrt(np.finfo(float).eps), k)
    
    err = check_grad(func_ln_nb, grad_ln_nb, x0, k) 

    print(grad)
    print(approx_grad)
    
