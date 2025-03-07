import numpy as np

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
