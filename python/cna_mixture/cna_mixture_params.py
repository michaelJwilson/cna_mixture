import logging

import numpy as np

from dataclasses import dataclass, field
from cna_mixture.cna_emission import get_ln_state_emission
from cna_mixture.plotting import plot_rdr_baf_flat

logger = logging.getLogger(__name__)


@dataclass
class CNA_mixture_params:
    num_cna_states: int = 3
    num_states: int = None
    rdr_overdispersion: float = 2.0e-2
    baf_overdispersion: float = 50.0
    normal_state: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.5]))
    cna_states: np.ndarray = field(default=None)

    def __post_init__(self):
        self.num_states = 1 + self.num_cna_states
        self.verify()

    def verify(self):
        msg = f"Inconsistent number of (CNA) states for:\n{self}"
        assert self.num_states == (self.num_cna_states + 1), msg

        if self.cna_states is not None:
            assert isinstance(
                self.cna_states, np.ndarray
            ), f"cna_states attribute must be a numpy array. Found {type(self.cna_states)}"

    def __setattr__(self, key, value):
        if key in self.__annotations__ or key in self.__dict__:
            super().__setattr__(key, value)
        else:
            raise AttributeError(
                f"Cannot set ill-defined attribute '{key}' for CNA_mixture_params"
            )

    def __str__(self):
        return ",  ".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    @property
    def params(self):
        if self.cna_states is None:
            raise ValueError("cna_states is not set.")

        return np.array(
            [
                *self.cna_states[:, 0].tolist(),
                self.rdr_overdispersion,
                *self.cna_states[:, 1].tolist(),
                self.baf_overdispersion,
            ]
        )

    def dict_update(self, input_params_dict):
        """
        Update an instance of CNA_mixture_params to the input key: value dict.

        Assumes input dictionary specifies all cna mixture attributes.
        """
        params_dict = input_params_dict.copy()
        keys = list(params_dict.keys())

        for key in keys:
            value = params_dict[key]

            setattr(self, key, value)

            # NB fails if input_params_dict missing required key.
            params_dict.pop(key)

        if params_dict:
            logger.warning(f"Skipping additional params in provided dict={params_dict}")

        self.cna_states = np.array(self.cna_states)

        self.num_cna_states = len(self.cna_states)
        self.num_states = 1 + self.num_cna_states

        self.verify()
