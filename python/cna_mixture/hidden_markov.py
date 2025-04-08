import numpy as np
from numba import njit

from cna_mixture.utils import logmatexp


@njit
def forward(ln_start_prior, transfer, ln_state_emission):
    # NB (# segments, # states)
    ln_fs = np.zeros_like(ln_state_emission)

    # NB forward start == categorical (where lambdas given by backward)?
    ln_fs[0, :] = ln_start_prior + ln_state_emission[0, :]

    for ii in range(1, len(ln_state_emission)):
        ln_fs[ii, :] = ln_state_emission[ii, :] + logmatexp(
            transfer, ln_fs[ii - 1, :].T
        )

    return ln_fs


@njit
def backward(ln_start_prior, transfer, ln_state_emission):
    # NB (# segments, # states)
    ln_bs = np.zeros_like(ln_state_emission)

    # NB forward start == categorical (where lambdas given by forward)?
    ln_bs[-1, :] = ln_start_prior

    # NB starts with element preceeding the last.
    for ii in range(len(ln_state_emission) - 2, -1, -1):
        # TODO BUG? assumes transfer matrix is symmetric, e.g. 3->2 == 2->3.
        ln_bs[ii, :] = logmatexp(
            transfer.T, ln_bs[ii + 1, :] + ln_state_emission[ii + 1, :]
        )

    return ln_bs


def agnostic_transfer(num_states, jump_rate):
    jump_rate_per_state = jump_rate / (num_states - 1.0)

    transfer = jump_rate_per_state * np.ones(shape=(num_states, num_states))

    transfer -= jump_rate_per_state * np.eye(num_states)
    transfer += (1.0 - jump_rate) * np.eye(num_states)

    return transfer


class CNA_transfer:
    def __init__(self, jump_rate=0.1, num_states=4):
        self.jump_rate = jump_rate
        self.num_states = num_states
        self.jump_rate_per_state = self.jump_rate / (self.num_states - 1.0)

        self.transfer_matrix = agnostic_transfer(self.num_states, self.jump_rate)
