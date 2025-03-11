import logging
import numpy as np
from scipy.special import logsumexp

from cna_mixture.transfer import CNA_transfer
from cna_mixture.utils import assign_closest, logmatexp, normalize_ln_probs
from cna_mixture_rs.core import ln_transition_probs_rs

logger = logging.getLogger()


class CNA_categorical_prior:
    def __init__(self, num_segments, num_states):
        self.num_segments = num_segments
        self.num_states = num_states

    def __str__(self):
        return f"lambdas={np.exp(self.ln_lambdas)}"

    def ln_lambdas_equal(self):
        self.ln_lambdas = np.log((1.0 / self.num_states) * np.ones(self.num_states))

    def ln_lambdas_closest(self, rdr_baf, cna_states):
        assert len(cna_states) == self.num_states

        decoded_states = assign_closest(rdr_baf, cna_states)

        # NB categorical prior on state fractions
        ustates, counts = np.unique(decoded_states, return_counts=True)

        counts = dict(zip(ustates, counts))
        counts = [counts.get(ii, 0) for ii in range(self.num_states)]

        # NB i.e. ln_lambdas
        self.ln_lambdas = np.log(counts) - np.log(np.sum(counts))

    def get_ln_state_priors(self):
        ln_norm = logsumexp(self.ln_lambdas)

        return np.broadcast_to(
            self.ln_lambdas - ln_norm, (self.num_segments, len(self.ln_lambdas))
        ).copy()

    def get_ln_state_posteriors(self, ln_state_emission):
        ln_state_prior = self.get_ln_state_priors()

        return normalize_ln_probs(ln_state_emission + ln_state_prior)

    def initialize(self, *args, **kwargs):
        self.ln_lambdas_closest(*args, **kwargs)

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


class CNA_markov_prior:
    def __init__(self, num_segments, num_states):
        self.num_segments = num_segments
        self.num_states = num_states

        self.ln_fs = np.zeros(shape=(num_segments, self.num_states))
        self.ln_bs = np.zeros(shape=(num_segments, self.num_states))

    def initialize(self, jump_rate, ln_start_prior=None):
        if ln_start_prior is None:
            ln_start_prior = np.log((1.0 / self.num_states) * np.ones(self.num_states))

        self.ln_start_prior = ln_start_prior
        self.transfer = CNA_transfer(
            jump_rate=jump_rate, num_states=self.num_states
        ).transfer_matrix

    def update(self, ln_state_emission):
        logger.warning("CNA_markov_prior.update is *not* implemented.")

        new_ln_transfer = np.array(
            ln_transition_probs_rs(
                self.num_states,
                self.ln_fs,
                self.ln_bs,
                np.log(self.transfer),
                ln_state_emission,
            )
        )

        self.transfer = np.exp(new_ln_transfer)

    def get_ln_state_priors(self):
        self.ln_fs[0, :] = self.ln_start_prior

        for ii in range(1, self.num_segments):
            self.ln_fs[ii, :] = logmatexp(self.transfer, self.ln_fs[ii - 1, :].T)

        return self.ln_fs

    def forward(self, ln_state_emission):
        self.ln_fs[0, :] = self.ln_start_prior + ln_state_emission[0, :]

        for ii in range(1, self.num_segments):
            self.ln_fs[ii, :] = ln_state_emission[ii, :]
            self.ln_fs[ii, :] += logmatexp(self.transfer, self.ln_fs[ii - 1, :].T)

    def backward(self, ln_state_emission):
        self.ln_bs[-1, :] = self.ln_start_prior

        for ii in range(self.num_segments - 2, -1, -1):
            self.ln_bs[ii, :] = logmatexp(
                self.transfer.T, self.ln_bs[ii + 1, :] + ln_state_emission[ii + 1, :]
            )

    def get_ln_state_posteriors(self, ln_state_emission, *args, **kwargs):
        self.forward(ln_state_emission)
        self.backward(ln_state_emission)

        norm = logsumexp(self.ln_fs + self.ln_bs, axis=1)

        return -norm[:, None] + (self.ln_fs + self.ln_bs)
