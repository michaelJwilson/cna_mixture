import numpy as np

def reparameterize_nbinom(means, overdisp):
    """
    Reparameterize negative binomial from per-state means
    and shared overdispersion to (num_successes, prob. of success).
    """
    # NB https://en.wikipedia.org/wiki/Negative_binomial_distribution.
    means = np.array(means)

    # NB [0.0, 1.0] by definition.
    ps = 1.0 / (1.0 + overdisp * means)

    # NB for overdisp << 1, r >> 1, Gamma(r) -> Stirling's / overflow.
    rs = np.ones_like(means) / overdisp

    return np.c_[rs, ps]

def poisson_state_logprobs(state_mus, ks):
    """
    log PDF for a Poisson distribution of given
    means and realized ks.
    """
    result = np.zeros((len(ks), len(state_mus)))

    for col, mu in enumerate(state_mus):
        for row, kk in enumerate(ks):
            result[row, col] = poisson.logpmf(kk, mu)

    return result
