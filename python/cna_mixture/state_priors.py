import numpy as np
from scipy.special import logsumexp

from cna_mixture.transfer import CNA_transfer
from cna_mixture.utils import assign_closest


class CNA_categorical_prior:
    def __init__(self, mixture_params, rdr_baf):
        self.num_states = mixture_params.num_states
        self.cna_states = mixture_params.cna_states
        self.ln_lambdas = self.ln_lambdas_closest(rdr_baf, self.cna_states)

    @staticmethod
    def ln_lambdas_equal(num_states):
        return np.log((1.0 / num_states) * np.ones(num_states))

    @staticmethod
    def ln_lambdas_closest(rdr_baf, cna_states):
        decoded_states = assign_closest(rdr_baf, cna_states)

        # NB categorical prior on state fractions
        ustates, counts = np.unique(decoded_states, return_counts=True)

        counts = dict(zip(ustates, counts))
        counts = [counts.get(ii, 0) for ii in range(len(cna_states))]

        # NB i.e. ln_lambdas
        return np.log(counts) - np.log(np.sum(counts))

    def update(self, ln_state_posteriors):
        """
        Given updated ln_state_posteriors, calculate the updated ln_lambdas.
        """
        assert ln_state_posteriors.ndim == 2

        # HACK *slow* guard against being passed probabilities, instead of log probs.
        assert np.all(ln_state_posteriors <= 0.0)

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
    def __init__(self, num_segments, jump_rate, num_states, ln_start_prior=None):
        if ln_start_prior is None:
            ln_start_prior = np.log((1. / num_states) * np.ones(num_states))

        self.num_segments = num_segments
        self.ln_start_prior = ln_start_prior
        self.transfer = CNA_transfer(jump_rate=jump_rate, num_states=num_states).transfer_matrix

        self.ln_fs = np.zeros(shape=(num_segments, num_states))
        
    def forward(self, ln_state_emission):
        self.ln_fs[0,:] = self.ln_start_prior + ln_state_emission[0,:]

        for ii in range(1, self.num_segments):
            self.ln_fs[ii,:] = np.log(np.dot(self.transfer, np.exp(self.ln_fs[ii-1,:]).T))
            self.ln_fs[ii,:] += ln_state_emission[ii,:]
            
    def backward(self, ln_state_emission):
        raise NotImplementedError()
        
    def get_state_priors(self, ln_state_emission):
        raise NotImplementedError()
