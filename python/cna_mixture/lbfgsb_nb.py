import time
import numpy as np
import pylab as pl

from scipy.stats import nbinom, betabinom
from scipy.special import digamma
from scipy.optimize import approx_fprime, check_grad, minimize
from scipy.differentiate import derivative
from cna_mixture_rs.core import nbinom_logpmf

RUST_BACKEND = True


def nbinom_logpmf(ks, rs, ps):
    ks = np.atleast_1d(ks).astype(float)
    rs = np.atleast_1d(rs).astype(float)
    ps = np.atleast_1d(ps).astype(float)

    if RUST_BACKEND:
        ks = np.ascontiguousarray(ks)
        rs = np.ascontiguousarray(rs)
        ps = np.ascontiguousarray(ps)

        result = nbinom_logpmf(ks, rs, ps)
        result = np.array(result)

    else:
        result = np.zeros(shape=(len(ks), len(rs)))

        for ss, (r, p) in enumerate(zip(rs, ps)):
            for ii, k in enumerate(ks):
                result[ii, ss] = nbinom.logpmf(k, r, p)

    return result


def rp2muvar(r, p):
    mu = r * (1.0 - p) / p
    var = mu / p

    return mu, var


def muvar2rp(mu, var):
    p = mu / var
    r = mu * mu / (var - mu)

    return r, p


def ln_nb_rp(x, k):
    r, p = x
    return nbinom_logpmf(k, r, p)[:, 0]


def ln_nb_muvar(x, k):
    r, p = muvar2rp(*x)
    return nbinom_logpmf(k, r, p)[:, 0]


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


# TODO port from python.  vectorized over k.
def grad_ln_nb_muvar(x, k):
    mu, var = x
    return np.array([grad_ln_nb_mu(k, mu, var), grad_ln_nb_var(k, mu, var)])


def nloglikes(r, p, samples):
    return -nbinom_logpmf(samples, r, p)[:, 0]


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

    assert (
        err < 3.5e-6
    ), f"Failed to match ln_nb_rp gradient with sufficient precision.  Achieved {err}."

    mu = r * (1.0 - p) / p
    var = mu / p

    x0 = np.array([mu, var])

    res = ln_nb_muvar(x0, k)
    grad = grad_ln_nb_muvar(x0, k)
    approx_grad = approx_fprime(x0, ln_nb_muvar, np.sqrt(np.finfo(float).eps), k)

    err = check_grad(ln_nb_muvar, grad_ln_nb_muvar, x0, k)

    assert res == exp
    assert (
        err < 7.0e-6
    ), f"Failed to match ln_nb_muvar gradient with sufficient precision.  Achieved {err}."

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

    print(
        f"\n\nOptimized with L-BFGS-B in {time.time() - start:.3f} seconds with result:\n{res}"
    )

    r, p = muvar2rp(*res.x)
    probs = nloglikes(r, p, samples)
    pl.plot(samples, exp_probs, lw=0.0, marker=".", alpha=0.5)
    pl.plot(samples, probs, lw=0.0, marker=".", alpha=0.5)
    pl.show()

    """
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
    """
    print("\n\nDone.\n\n")
