import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from scipy.stats import norm

np.random.seed(314)

def get_cost(samples, centers, scale=10.):
    cost = np.inf * np.ones_like(samples)
    
    for cc in np.atleast_1d(centers):
        new = -norm.logpdf(samples, loc=cc)
        cost = np.minimum(cost, new)

    return cost
        
def random_centers(samples, k=5):
    return np.random.choice(samples, replace=False, size=5)

def kmeans_plusplus(samples, k=5, scale=10.):
    idx = np.arange(len(samples))
    information = np.ones_like(samples)
    centers = []

    while len(centers) < k:
      ps = information / information.sum()
        
      # NB high exclusive, with replacement.                                                                                                                                                                       
      xx = samples[np.random.choice(idx, p=ps)]
      centers.append(xx)

      information = get_cost(samples, centers)

    return information.sum(), np.array(centers)

def norm_sim(num_components=5):
    scale = 100.
    num_samples = 500_000

    mus = scale * np.sort(np.random.uniform(size=num_components))
    sigmas = (scale / 10.) * np.random.uniform(size=num_components)

    lambdas = np.random.uniform(size=num_components)
    lambdas /= lambdas.sum()
    
    print(mus)
    print(sigmas)
    print(lambdas)

    cluster_idx = np.arange(num_components)
    cluster_samples = np.random.choice(cluster_idx, p=lambdas, size=num_samples)

    samples = []

    for idx in cluster_samples:
        samples.append(
            np.random.normal(loc=mus[idx], scale=sigmas[idx])
        )

    samples = np.array(samples)

    return samples, lambdas, mus, sigmas

def plot_sim(samples, centers, lambdas, mus, sigmas):
    pl.hist(samples, bins=np.arange(0., 100., 1.), histtype='step', density=True)

    xs = np.linspace(0.0, 100., 1000)

    for ll, mu, sigma in zip(lambdas, mus, sigmas):
        ys = norm.pdf(xs, mu, sigma)
        plt.plot(xs, ll * ys)

    for xx in centers:
        pl.axvline(xx, c="k", lw=0.5)
        
    pl.xlabel(r"$x$")
    pl.ylabel(r"$P(x)$")
    pl.show()


if __name__ == "__main__":
    samples, lambdas, mus, sigmas = norm_sim()
    
    cost, centers = kmeans_plusplus(samples)

    plot_sim(samples, centers, lambdas, mus, sigmas)
