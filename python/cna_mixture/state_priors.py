import logging

import numpy as np
from cna_mixture_rs.core import ln_transition_probs_rs
from numba import njit
from scipy.special import logsumexp

from cna_mixture.transfer import CNA_transfer
from cna_mixture.utils import assign_closest, logmatexp, normalize_ln_probs

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

    def initialize(self, **kwargs):
        self.ln_lambdas_closest(kwargs["rdr_baf"], kwargs["cna_states"])

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


@njit
def forward(ln_start_prior, transfer, ln_state_emission):
    ln_fs = np.zeros_like(ln_state_emission)
    ln_fs[0, :] = ln_start_prior + ln_state_emission[0, :]

    for ii in range(1, len(ln_state_emission)):
        ln_fs[ii, :] = ln_state_emission[ii, :]
        ln_fs[ii, :] += logmatexp(transfer, ln_fs[ii - 1, :].T)

    return ln_fs


@njit
def backward(ln_start_prior, transfer, ln_state_emission):
    ln_bs = np.zeros_like(ln_state_emission)
    ln_bs[-1, :] = ln_start_prior

    for ii in range(len(ln_state_emission) - 2, -1, -1):
        ln_bs[ii, :] = logmatexp(
            transfer.T, ln_bs[ii + 1, :] + ln_state_emission[ii + 1, :]
        )

    return ln_bs


class CNA_markov_prior:
    def __init__(self, num_segments, num_states):
        self.num_segments = num_segments
        self.num_states = num_states

    def initialize(self, **kwargs):
        if "ln_start_prior" in kwargs:
            self.ln_start_prior = kwargs["ln_start_prior"]
        else:
            self.ln_start_prior = np.log((1.0 / self.num_states) * np.ones(self.num_states))

        if "jump_rate" in kwargs:
            jump_rate=kwargs["jump_rate"]
        else:
            jump_rate=0.1
            
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

    # TODO HACK
    def get_ln_state_priors(self):
        return forward(
            self.ln_start_prior,
            self.transfer,
            np.zeros(shape=(self.num_segments, self.num_states)),
        )

    def get_ln_state_posteriors(self, ln_state_emission, *args, **kwargs):
        self.ln_fs = forward(self.ln_start_prior, self.transfer, ln_state_emission)
        self.ln_bs = backward(self.ln_start_prior, self.transfer, ln_state_emission)

        norm = logsumexp(self.ln_fs + self.ln_bs, axis=1)

        return -norm[:, None] + (self.ln_fs + self.ln_bs)
