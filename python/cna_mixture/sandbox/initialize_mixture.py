
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from scipy.stats import norm


np.random.seed(314)

scale = 100.
num_components, num_samples = 5, 500_000

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

pl.hist(samples, bins=np.arange(0., 100., 1.), histtype='step', density=True)

xs = np.linspace(0.0, 100., 1000)

for ll, mu, sigma in zip(lambdas, mus, sigmas):
    ys = norm.pdf(xs, mu, sigma)    
    plt.plot(xs, ll * ys)

pl.xlabel(r"$x$")
pl.ylabel(r"$P(x)$")
pl.show()
