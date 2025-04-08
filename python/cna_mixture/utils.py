import logging

import numpy as np
from numba import njit
from scipy.spatial import KDTree
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


def uniform_ln_probs(num_states):
    return np.log((1.0 / num_states) * np.ones(num_states))


def normalize_ln_probs(ln_probs):
    """
    Return the normalized log probs (# samples, # states).
    """
    num_samples, num_states = ln_probs.shape

    # NB natural logarithm by definition;
    norm = logsumexp(ln_probs, axis=1)
    norm = np.broadcast_to(norm.reshape(num_samples, 1), (num_samples, num_states))

    return ln_probs.copy() - norm


def param_diff(params, new_params):
    if (params is None) or (new_params is None):
        return np.inf

    return np.max(np.abs(1.0 - new_params / params))


def assign_closest(points, centers):
    """
    Assign points to the closest center.
    """
    if len(centers) > len(points):
        logger.warning(
            f"Expected more centers than points, found {len(centers)} and {len(points)} respectively."
        )

    tree = KDTree(centers)
    distances, idx = tree.query(points)

    return idx


@njit
def logmatexp(transfer, ln_probs):
    max_ln_probs = np.max(ln_probs)
    return max_ln_probs + np.log(np.dot(transfer, np.exp(ln_probs - max_ln_probs)))
