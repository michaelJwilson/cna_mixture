import logging

import numpy as np
from cna_mixture_rs.core import ln_transition_probs_rs
from scipy.special import logsumexp

from cna_mixture.hidden_markov import CNA_transfer, backward, forward
from cna_mixture.utils import assign_closest, normalize_ln_probs

logger = logging.getLogger()


class CNA_categorical_prior:
    def __init__(self, num_segments, num_states, production_mode=True):
        logger.info(
            f"Initializing CNA_categorical_prior for num. segments, num. states = {num_segments}, {num_states} respectively."
        )

        self.num_states = num_states
        self.num_segments = num_segments
        self.production_mode = production_mode

    def __str__(self):
        return f"lambdas={np.exp(self.ln_lambdas)}"

    def ln_lambdas_equal(self):
        """
        Initialize categorical prior probs. (lambdas) to be equal.
        """
        self.ln_lambdas = np.log((1.0 / self.num_states) * np.ones(self.num_states))

    def ln_lambdas_closest(self, rdr_baf, cna_states):
        """
        Initialize categorical prior probs. (lambdas) according to
        an assignment of each (RDR, BAF) point to it's nearest current state.
        """
        assert len(cna_states) == self.num_states

        decoded_states = assign_closest(rdr_baf, cna_states)

        # NB categorical prior on state fractions
        ustates, counts = np.unique(decoded_states, return_counts=True)

        counts = dict(zip(ustates, counts, strict=False))
        counts = [counts.get(ii, 0) for ii in range(self.num_states)]

        # NB i.e. normalized ln_lambdas.
        self.ln_lambdas = np.log(counts) - np.log(np.sum(counts))

    def initialize(self, **kwargs):
        logger.info(
            "Initializing Categorical state prior with lambdas defined by nearest state assignment"
        )

        self.ln_lambdas_closest(kwargs["rdr_baf"], kwargs["cna_states"])

    def get_ln_state_priors(self):
        ln_norm = logsumexp(self.ln_lambdas)

        return np.broadcast_to(
            self.ln_lambdas - ln_norm, (self.num_segments, len(self.ln_lambdas))
        ).copy()

    def get_ln_state_posteriors(self, ln_state_emission):
        """
        Given the log. state emission probability, return the ln state
        posteriors.
        """
        ln_state_prior = self.get_ln_state_priors()

        return normalize_ln_probs(ln_state_emission + ln_state_prior)

    def update(self, ln_state_posteriors):
        """
        Given updated ln_state_posteriors, calculate the updated ln_lambdas for a
        Categorical model.
        """
        assert ln_state_posteriors.ndim == 2

        # NB *slow* guard against being passed probabilities, instead of log probs.
        if not self.production_mode:
            assert np.all(ln_state_posteriors <= 0.0)

        self.ln_lambdas = logsumexp(ln_state_posteriors, axis=0) - logsumexp(
            ln_state_posteriors
        )


class CNA_markov_prior:
    def __init__(self, num_segments, num_states, seed=314):
        logger.info(
            f"Initializing CNA_markov_prior for num. segments, num. states = {num_segments}, {num_states} respectively."
        )

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.num_segments = num_segments
        self.num_states = num_states

    def initialize(self, **kwargs):
        self.ln_start_prior = kwargs.get(
            "ln_start_prior", np.log((1.0 / self.num_states) * np.ones(self.num_states))
        )

        self.jump_rate = kwargs.get("jump_rate", 0.1)
        self.transfer = CNA_transfer(
            jump_rate=self.jump_rate, num_states=self.num_states
        ).transfer_matrix

    def sample_hidden(self):
        start_prior = np.exp(self.ln_start_prior)
        state = self.rng.choice(
            np.arange(self.num_states), size=1, p=start_prior
        )[0]

        result = [state]
        
        for ii in range(self.num_segments -1):
            transfer_probs = self.transfer[state]
            state = self.rng.choice(
                np.arange(self.num_states), size=1, p=transfer_probs
            )[0]

            result.append(state)
            
        return np.array(result)
            
    def update(self, ln_state_emission):
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
        """
        Returns a zeros-array, i.e. delegating solely to emission prob.
        """
        # TODO check this is appropriate?
        return (np.zeros(shape=(self.num_segments, self.num_states)),)

    # TODO outlier masking?
    def get_ln_state_posteriors(self, ln_state_emission):
        self.ln_fs = forward(self.ln_start_prior, self.transfer, ln_state_emission)
        self.ln_bs = backward(self.ln_start_prior, self.transfer, ln_state_emission)

        norm = logsumexp(self.ln_fs + self.ln_bs, axis=1)

        return -norm[:, None] + (self.ln_fs + self.ln_bs)
