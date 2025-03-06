import numpy as np
from scipy.special import logsumexp
from scipy.spatial import KDTree


def normalize_ln_posteriors(ln_posteriors):
    """
    Return the normalized log posteriors.
    """
    num_samples, num_states = ln_posteriors.shape

    # NB natural logarithm by definition;
    norm = logsumexp(ln_posteriors, axis=1)
    norm = np.broadcast_to(norm.reshape(num_samples, 1), (num_samples, num_states))

    return ln_posteriors.copy() - norm


def param_diff(params, new_params):
    if params is None:
        return np.inf
    elif new_params is None:
        return np.inf
    else:
        return np.max(np.abs((1.0 - new_params / params)))


def assign_closest(points, centers):
    """
    Assign points to the closest center.
    """
    assert len(points) > len(centers)

    tree = KDTree(centers)
    distances, idx = tree.query(points)

    return idx
