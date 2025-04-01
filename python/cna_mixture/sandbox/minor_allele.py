import numpy as np
import pylab as pl
from scipy.stats import betabinom

"""
A study of the consequences of minor allele definition for a NB/BAF
model.
"""

pp = 0.45
tau = 10_000
num_trials = 100
num_samples = 250_000

alpha = pp * tau
beta = (1.0 - pp) * tau

pl.axvline(pp * num_trials, c="k", lw=0.5)

samples = np.array(
    [betabinom.rvs(num_trials, alpha, beta) for ii in range(num_samples)]
)

pl.hist(
    samples,
    bins=np.arange(0, num_trials, 1),
    histtype="step",
    density=True,
    label=r"$b$-allele, " + f"$\mu={pp:.2f}$",
)

idx = np.where(samples > 50)
samples[idx] = num_trials - samples[idx]

pl.hist(
    samples,
    bins=np.arange(0, num_trials, 1),
    histtype="step",
    density=True,
    label="minor allele (folded)",
)

pl.xlabel(f"reads | {num_trials} coverage")
pl.legend(frameon=False)
pl.show()
