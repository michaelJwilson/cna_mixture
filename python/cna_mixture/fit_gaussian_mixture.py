import logging
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from cna_mixture.encoding import onehot_encode_states
from cna_mixture.plotting import plot_rdr_baf_flat

logger = logging.getLogger(__name__)


def fit_gaussian_mixture(
    fpath,
    rdr_baf,
    num_samples=100_000,
    num_components=4,
    max_iter=1,
    covariance_type="diag",
    seed=314,
):
    """
    See:  https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/src/calicost/utils_hmm.py#L163    
    """
    logger.info(
        f"Fitting Gaussian mixture model with {num_components} components and {covariance_type} covariance, {max_iter} max. iterations."
    )
    
    # NB covariance_type = {diag, full}; see https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    # NB catch ill-convergence warning due to max_iter=1. 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        gmm = GaussianMixture(
            n_components=num_components,
            max_iter=max_iter,
            covariance_type=covariance_type,
            random_state=seed,
        ).fit(rdr_baf)

    means = np.c_[gmm.means_[:, 0], gmm.means_[:, 1]]
    samples, decoded_states = gmm.sample(n_samples=num_samples)

    logger.info(f"Found Gaussian-mixture means:\n{means}")

    # NB catch log(0) warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        ln_state_posteriors = np.log(onehot_encode_states(decoded_states))

    plot_rdr_baf_flat(
        fpath,
        samples[:, 0],
        samples[:, 1],
        ln_state_posteriors=ln_state_posteriors,
        states_bag=means,
        title=r"Best-fit Gaussian Mixture Model samples",
    )
