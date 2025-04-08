import numpy as np
from numba import njit


class CNA_transfer:
    def __init__(self, jump_rate=0.1, num_states=4):
        self.jump_rate = jump_rate
        self.num_states = num_states
        self.jump_rate_per_state = self.jump_rate / (self.num_states - 1.0)

        self.transfer_matrix = self.jump_rate_per_state * np.ones(
            shape=(self.num_states, self.num_states)
        )
        self.transfer_matrix -= self.jump_rate_per_state * np.eye(self.num_states)
        self.transfer_matrix += (1.0 - self.jump_rate) * np.eye(self.num_states)


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
