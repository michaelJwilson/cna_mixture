import numpy as np

def onehot_encode_states(state_array):
    """
    Given an index array of categorical states,
    return the (# samples, # states) one-hot
    encoding.

    NB equivalent to a (singular) state posterior!
    """
    num_states = np.max(state_array).astype(int) + 1
    states = state_array.astype(int)

    return np.eye(num_states)[states]
