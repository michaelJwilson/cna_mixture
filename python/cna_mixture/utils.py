import numpy as np
from scipy.special import logsumexp

def normalize_ln_posteriors(ln_posteriors):
    """
    Return the normalized log posteriors.
    """
    num_samples, num_states = ln_posteriors.shape

    # NB natural logarithm by definition;
    norm = logsumexp(ln_posteriors, axis=1)
    norm = np.broadcast_to(norm.reshape(num_samples, 1), (num_samples, num_states))

    return ln_posteriors.copy() - norm
