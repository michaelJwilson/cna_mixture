import logging
import functools

import numpy as np
from numba import njit
from scipy.spatial import KDTree
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


def deprecated(reason: str):
    """
    A decorator to mark functions or methods as deprecated.

    Raises a RuntimeError with the provided reason when the function is called.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            raise RuntimeError(
                f"The function '{func.__name__}' is deprecated: {reason}"
            )

        return wrapper

    return decorator


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


def patch_default(values, default):
    # NB assumes one dimension.
    valid = np.isfinite(values)

    result = values.copy()
    result[~valid] = default

    return result


def assign_closest(points, centers):
    """
    Assign points to the closest center.
    """
    if len(centers) > len(points):
        logger.warning(
            f"Expected more centers than points, found {len(centers)} and {len(points)} respectively."
        )

    # TODO warn on masking.
    valid = np.isfinite(points)
    valid = np.all(valid, axis=1)

    tree = KDTree(centers)
    distances, idx = tree.query(points[valid])

    return idx


@njit
def logmatexp(transfer, ln_probs):
    max_ln_probs = np.max(ln_probs)
    return max_ln_probs + np.log(np.dot(transfer, np.exp(ln_probs - max_ln_probs)))


@njit
def cosine_similarity_origin(array):
    norm = np.linalg.norm(array[0])
    result = []

    for row in array:
        row_norm = np.linalg.norm(row)
        result.append(np.dot(row, array[0]) / norm / row_norm)

    return np.array(result)
