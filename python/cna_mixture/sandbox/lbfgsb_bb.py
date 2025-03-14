import time

import numpy as np
from cna_mixture_rs.core import betabinom_logpmf as betabinom_logpmf_rs
from scipy.optimize import approx_fprime, check_grad, minimize
from scipy.special import digamma
from scipy.stats import betabinom

RUST_BACKEND = False

np.random.seed(1234)

epsilon = np.sqrt(np.finfo(float).eps)


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

        result = betabinom_logpmf_rs(ks, ns, alphas, betas)
        result = np.array(result)
    else:
        result = np.zeros(shape=(len(ks), len(alphas)))

        for ss, (a, b) in enumerate(zip(alphas, betas)):
            for ii, (k, n) in enumerate(zip(ks, ns)):
                result[ii, ss] = betabinom.logpmf(k, n, a, b)

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


def nloglikes_ab(x, ks, ns):
    return -betabinom_logpmf(ks, ns, *x)[:, 0]


def nloglike_ab(x, ks, ns):
    return nloglikes_ab(x, ks, ns).sum()


def nloglikes_pt(x, ks, ns):
    a, b = pt2ab(*x)
    return -betabinom_logpmf(ks, ns, a, b)[:, 0]


def nloglike_pt(x, ks, ns):
    return nloglikes_pt(x, ks, ns).sum()


def grad_nloglike_pt(x, ks, ns):
    result = np.zeros(2)

    for k, n in zip(ks, ns):
        result += grad_ln_bb_pt(x, k, n)

    return -result


if __name__ == "__main__":
    nsample = 10_000
    alphas, betas = np.array([0.6]), np.array([0.4])

    ns = np.random.randint(low=25, high=500, size=nsample)
    ks = np.array([betabinom.rvs(n, alphas[0], betas[0]) for n in ns])

    x0 = (0.9, 0.1)
    p0 = ab2pt(*x0)

    res = nloglike_ab(x0, ks, ns)
    grad = grad_ln_bb_ab(x0, ks[0], ns[0])
    approx_grad = approx_fprime(
        x0, ln_bb_ab, np.sqrt(np.finfo(float).eps), ks[0], ns[0]
    )

    err = check_grad(ln_bb_ab, grad_ln_bb_ab, x0, ks[0], ns[0])

    assert (
        err < 2.0e-5
    ), f"Failed to match ln_bb_ab gradient with sufficient precision.  Achieved {err}."

    grad = grad_ln_bb_pt(p0, ks[0], ns[0])
    approx_grad = approx_fprime(
        p0, ln_bb_pt, np.sqrt(np.finfo(float).eps), ks[0], ns[0]
    )

    err = check_grad(ln_bb_pt, grad_ln_bb_pt, x0, ks[0], ns[0])

    assert (
        err < 1.51e-5
    ), f"Failed to match ln_bb_ab gradient with sufficient precision.  Achieved {err}."

    exp = ab2pt(alphas[0], betas[0])

    print(f"\nExpectation={exp}\n")

    ## >>>>  L-BFGS-B analytic gradients.                                                                                                                                                                           
    start = time.time()
    bounds = [(epsilon, 1.0), (epsilon, None)]
    res = minimize(
        nloglike_pt,
        p0,
        args=(ks, ns),
        method="L-BFGS-B",
        jac=grad_nloglike_pt,
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
        
    ## >>>>  L-BFGS-B no analytic gradients.
    start = time.time()
    bounds = [(epsilon, 1.0), (epsilon, None)]
    res = minimize(
        nloglike_pt,
        p0,
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

    ## >>>>  Powell's
    start = time.time()
    bounds = [(epsilon, 1.0), (epsilon, None)]
    res = minimize(
        nloglike_pt,
        p0,
	args=(ks, ns),
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

    print(
        f"\n\nOptimized with Powell's in {time.time() - start:.3f} seconds with result:\n{res}"
    )
    
    ## >>>>  Nelder-Mead
    start = time.time()
    bounds = [(epsilon, 1.0), (epsilon, None)]
    res = minimize(
        nloglike_pt,
        p0,
	args=(ks, ns),
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

    print(
        f"\n\nOptimized with Nelder-Mead in {time.time() - start:.3f} seconds with result:\n{res}"
    )
    
    print("\n\nDone.\n\n")
