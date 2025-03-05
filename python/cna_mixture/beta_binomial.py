import numpy as np

def reparameterize_beta_binom(input_bafs, overdispersion):
    """
    Given the array of BAFs for all states and a shared overdispersion,
    return the (# states, 2) array of [alpha, beta] for each state,
    where beta is associated to the BAF probability.
    """
    return np.array(
        [
            [
                (1.0 - baf) * overdispersion,
                baf * overdispersion,
            ]
            for baf in input_bafs
        ]
    )
