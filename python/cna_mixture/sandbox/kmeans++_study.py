import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from numba import njit
from scipy.stats import norm
from multiprocessing import Pool

"""
A study of kmeans++ like assignments for a 1D Gaussian mixture simulation.
"""

np.random.seed(314)

NUM_WORKERS = 8


@njit
def norm_logpdf(xs, mu, sigma):
    """
    Log probability for the normal distribution.
    """
    return -0.5 * ((xs - mu) / sigma) ** 2.0 - 0.5 * np.log(2.0 * np.pi) - np.log(sigma)


@njit
def norm_entropy(sigma):
    """
    Entropy for the normal distribution.

    See:  https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
    """
    return 0.5 * (1.0 + np.log(2.0 * np.pi * sigma**2.0))


@njit
def get_cost(samples, centers, scale=10.0):
    """
    Return the kmeans++ cost, i.e. log pdf when each sample is matched to
    its nearest component.
    """
    cost = np.inf * np.ones_like(samples)

    for cc in centers:
        new = -norm_logpdf(samples, cc, scale)
        cost = np.minimum(cost, new)

    return cost


def random_centers(samples, k=5, scale=10.0):
    """
    Random centers (of degree k) and their associated cost.
    """
    centers = np.random.choice(samples, replace=False, size=k)
    cost = get_cost(samples, centers, scale=scale)

    return np.array(centers + [cost.sum()])


def kmeans_plusplus(samples, k=5, scale=10.0):
    """
    kmeans++ centers (of degree k) and their associated cost.

    NB samples new center according to -log prob. of
       existing centers.
    """
    idx = np.arange(len(samples))

    # NB -log Probability.
    information = np.ones_like(samples)
    centers = []

    while len(centers) < k:
        ps = information / information.sum()

        # NB high exclusive; with replacement.
        xx = samples[np.random.choice(idx, p=ps)]
        centers.append(xx)

        information = get_cost(samples, centers)

    return np.array(centers + [information.sum()])


def saturated_kmeans_plusplus(samples, k=5, scale=10.0):
    """
    Saturated kmeans++ centers (of degree k) and their associated cost.                                                                                                                                                                                                                                                                                                                               NB samples new center according to -log prob. of
       existing centers.
    """
    idx = np.arange(len(samples))

    # NB -log Probability.
    information = np.ones_like(samples) / len(samples)
    centers = []

    threshold = -np.log(1.0e-6)

    while len(centers) < k:
        ps = np.clip(information, a_min=0.0, a_max=threshold)
        ps /= ps.sum()

        # NB high exclusive; with replacement.
        xx = samples[np.random.choice(idx, p=ps)]
        centers.append(xx)

        information = get_cost(samples, centers)

    return np.array(centers + [information.sum()])


def greedy_kmeans_plusplus(samples, k=5, scale=10.0, N=4):
    """
    greedy sampling of kmeans++ centers (of degree k)
    and their associated cost.

    NB samples new center according to -log prob. of                                                                                                                                                                    existing centers.
    """
    idx = np.arange(len(samples))

    centers = []
    information = np.ones_like(samples)

    # NB
    # entropy_threshold = norm_entropy(scale)

    while len(centers) < k:
        ps = information.copy()
        ps /= ps.sum()

        # NB high exclusive, with replacement.
        size = N
        xs = samples[np.random.choice(idx, p=ps, size=size, replace=True)]

        costs = [get_cost(samples, centers + [xx]) for xx in xs]
        costs_sum = np.array([cost.sum() for cost in costs])

        minimizer = np.argmin(costs_sum)

        centers.append(xs[minimizer])

        information = costs[minimizer]

    return np.array(centers + [costs_sum[minimizer]])


def get_assignment_cost(assignment, samples, k=5, scale=10.0, maxiter=300):
    result = [assignment(samples) for _ in range(maxiter)]
    """
    # DEPRECATE BUG
    with Pool(NUM_WORKERS) as pool:
        args = (samples.copy() for _ in range(maxiter))
        result = pool.map(func, args)
    """
    return np.array(result)


def simulate_gaussian_mixture(num_components=5):
    """
    1D Gaussian mixture simultion with {num_components} centers.
    """
    num_samples, scale = 500_000, 100.0

    mus = scale * np.sort(np.random.uniform(size=num_components))
    sigmas = (scale / 10.0) * np.random.uniform(size=num_components)

    # NB responsibilities.
    lambdas = np.random.uniform(size=num_components)
    lambdas /= lambdas.sum()

    # NB simulate latent state variables.
    cluster_idx = np.arange(num_components)
    cluster_samples = np.random.choice(cluster_idx, p=lambdas, size=num_samples)

    # NB simulate emission model for each latent state.
    samples = np.array(
        [np.random.normal(loc=mus[idx], scale=sigmas[idx]) for idx in cluster_samples]
    )

    return lambdas, mus, sigmas, samples


def plot_gaussian_mixture(samples, centers, lambdas, mus, sigmas):
    """
    Plot 1D Gaussian mixture simultion with {num_components} centers.
    """
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

    lambdas, mus, sigmas, samples = simulate_gaussian_mixture()

    # TODO BUG scale is inaccurate.
    true_cost = get_cost(samples, mus, scale=10.0).sum()

    # plot_gaussian_mixture(samples, centers, lambdas, mus, sigmas)

    # NB a series of assignment techniques to study.
    assignments = [
        random_centers,
        kmeans_plusplus,
        # saturated_kmeans_plusplus,
        greedy_kmeans_plusplus,
    ]

    labels = ["random", "k++", "4-greedy k++"]

    for label, assignment in zip(labels, assignments):
        result = get_assignment_cost(assignment, samples)

        costs = result[:, -1] / len(samples)

        true = true_cost / len(samples)

        # NB kmeans++ bound.
        bound = 8.0 * (np.log(5.0) + 2.0) * true

        Ns = np.arange(1, 1 + len(result), 1)
        result = np.cumsum(costs) / Ns

        pl.plot(Ns[10:], result[10:], lw=0.5, label=label)

        mean = costs.mean()

    pl.ylim(3.3, 4.0)
    pl.xlabel("Realizations")
    pl.ylabel("Shannon Information [Nats]")
    # pl.axhline(mean, c="c", lw=0.5)
    pl.axhline(true, c="k", lw=0.5, label="truth")
    pl.title(
        f"<$\Phi$>={mean:.4f}; frac. error to truth={100. * (mean / true - 1.):.2f}%"
    )
    pl.legend(frameon=False)
    pl.show()
