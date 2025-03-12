import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from numba import njit
from scipy.stats import norm
from multiprocessing import Pool

np.random.seed(314)

NUM_WORKERS = 8


@njit
def norm_logpdf(xs, mu, sigma):
    return -0.5 * ((xs - mu) / sigma) ** 2.0 - 0.5 * np.log(2.0 * np.pi) - np.log(sigma)


@njit
def norm_entropy(sigma):
    return 0.5 * (1.0 + np.log(2.0 * np.pi * sigma**2.0))


@njit
def get_cost(samples, centers, scale=10.0):
    cost = np.inf * np.ones_like(samples)

    for cc in centers:
        new = -norm_logpdf(samples, cc, scale)
        cost = np.minimum(cost, new)

    return cost


def random_centers(samples, k=5, scale=10.0):
    centers = np.random.choice(samples, replace=False, size=k)
    cost = get_cost(samples, centers, scale=scale)

    return np.array(centers + [cost.sum()])
    

def kmeans_plusplus(samples, k=5, scale=10.0):
    idx = np.arange(len(samples))
    information = np.ones_like(samples)
    centers = []

    while len(centers) < k:
        ps = information / information.sum()

        # NB high exclusive, with replacement.
        xx = samples[np.random.choice(idx, p=ps)]
        centers.append(xx)

        information = get_cost(samples, centers)

    return np.array(centers + [information.sum()])


def mixture_plusplus(samples, k=5, scale=10.0, N=4):
    idx = np.arange(len(samples))

    centers = [samples[np.random.choice(idx)]]
    information = get_cost(samples, centers)

    # NB
    entropy_threshold = norm_entropy(scale)

    while len(centers) < k:
        ps = information.copy()
        # ps[information < entropy_threshold] = 0.0
        ps /= ps.sum()

        # NB high exclusive, with replacement.
        choice = np.random.choice(idx, p=ps, size=N)
        xs = samples[choice]

        interim = np.array([get_cost(samples, centers + [xx]).sum() for xx in xs])
        minimizer = np.argmin(interim)

        centers.append(xs[minimizer])

        information = get_cost(samples, centers)

    return np.array(centers + [information.sum()])


def initialize_exp(func, samples, k=5, scale=10.0, maxiter=5_000):
    # result = [func(samples) for _ in range(maxiter)]

    with Pool(NUM_WORKERS) as pool:
        args = (samples.copy() for _ in range(maxiter))
        result = pool.map(func, args)

    return np.array(result)


def norm_sim(num_components=5):
    scale = 100.0
    num_samples = 500_000

    mus = scale * np.sort(np.random.uniform(size=num_components))
    sigmas = (scale / 10.0) * np.random.uniform(size=num_components)

    lambdas = np.random.uniform(size=num_components)
    lambdas /= lambdas.sum()

    print(mus)
    print(sigmas)
    print(lambdas)

    cluster_idx = np.arange(num_components)
    cluster_samples = np.random.choice(cluster_idx, p=lambdas, size=num_samples)

    samples = []

    for idx in cluster_samples:
        samples.append(np.random.normal(loc=mus[idx], scale=sigmas[idx]))

    samples = np.array(samples)

    return samples, lambdas, mus, sigmas


def plot_sim(samples, centers, lambdas, mus, sigmas):
    pl.hist(samples, bins=np.arange(0.0, 100.0, 1.0), histtype="step", density=True)

    xs = np.linspace(0.0, 100.0, 1000)

    for ll, mu, sigma in zip(lambdas, mus, sigmas):
        ys = norm.pdf(xs, mu, sigma)
        plt.plot(xs, ll * ys)

    for xx in centers:
        pl.axvline(xx, c="k", lw=0.5)

    pl.xlabel(r"$x$")
    pl.ylabel(r"$P(x)$")
    pl.show()


if __name__ == "__main__":
    assert norm.logpdf(10.0, 1.0, 5.0) == norm_logpdf(10.0, 1.0, 5.0)

    samples, lambdas, mus, sigmas = norm_sim()
    true_cost = get_cost(samples, mus, scale=10.0).sum()

    # plot_sim(samples, centers, lambdas, mus, sigmas)

    result = initialize_exp(random_centers, samples)
    costs = result[:, -1] / len(samples)

    true = true_cost / len(samples)
    bound = 8.0 * (np.log(5.0) + 2.0) * true

    Ns = np.arange(1, 1 + len(result), 1)
    result = np.cumsum(costs) / Ns
    
    pl.plot(Ns, result, c="k", lw=0.5)
    pl.axhline(true, c="k", lw=0.5)
    pl.show()
