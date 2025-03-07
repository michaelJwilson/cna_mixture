import numpy as np
from scipy.special import logsumexp

from cna_mixture.utils import assign_closest


class CNA_categorical_prior:
    def __init__(self, mixture_params, rdr_baf):
        self.num_states = mixture_params.num_states
        self.cna_states = mixture_params.cna_states
        self.ln_lambdas = self.ln_lambdas_closest(rdr_baf, self.cna_states)

    @staticmethod
    def ln_lambdas_equal(num_states):
        return (1.0 / num_states) * np.ones(num_states)

    @staticmethod
    def ln_lambdas_closest(rdr_baf, cna_states):
        decoded_states = assign_closest(rdr_baf, cna_states)

        # NB categorical prior on state fractions
        _, counts = np.unique(decoded_states, return_counts=True)

        # NB i.e. ln_lambdas
        return np.log(counts) - np.log(np.sum(counts))

    def update(self, ln_state_posteriors):
        """
        Given updated ln_state_posteriors, calculate the updated ln_lambdas.
        """
        self.ln_lambdas = logsumexp(ln_state_posteriors, axis=0) - logsumexp(
            ln_state_posteriors
        )

    def get_state_priors(self, num_segments):
        """
        Broadcast per-state categorical priors to equivalent (samples x state)
        Prior array.
        """
        ln_norm = logsumexp(self.ln_lambdas)

        # NB ensure normalized.
        return np.broadcast_to(
            self.ln_lambdas - ln_norm, (num_segments, len(self.ln_lambdas))
        ).copy()

    def __str__(self):
        return f"lambdas={np.exp(self.ln_lambdas)}"
    
class CNA_markov_prior:
    def __init__(self):
        raise NotImplementedError()
