import logging

import numpy as np

from cna_mixture.cna_emission import get_ln_state_emission
from cna_mixture.plotting import plot_rdr_baf_flat

logger = logging.getLogger(__name__)


class CNA_mixture_params:
    """
    Data class for parameters required by CNA mixture model, with shared
    overdispersions.
    """

    def __init__(
        self, num_cna_states=3, phi=2.0e-2, tau=50.0
    ):
        """
        Initialize an instance of the class with random values in the assumed bounds.
        """
        # NB normal state is treated independently
        self.num_cna_states = num_cna_states
        self.num_states = 1 + self.num_cna_states

        # NB BAF overdispersion.  Random between 25. and 55.
        self.overdisp_tau = tau

        # NB RDR overdispersion.  Random between 1e-2 and 4e-2
        self.overdisp_phi = phi

        self.normal_state = np.array([1.0, 0.5])
        self.cna_states = None

    def __verify(self):
        assert isinstance(
            self.cna_states, np.ndarray
        ), f"cna_states attribute must be a numpy array. Found {type(self.cna_states)}"

    def __str__(self):
        return ",  ".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    @property
    def params(self):
        return np.array(
            [
                *self.cna_states[:, 0].tolist(),
                self.overdisp_phi,
                *self.cna_states[:, 1].tolist(),
                self.overdisp_tau,
            ]
        )
    
    def dict_update(self, input_params_dict):
        """
        Update an instance of CNA_mixture_params to the input key: value dict.

        Assumes input dictionary specifies all cna mixture attributes.

        """
        keys = self.__dict__.keys()
        params_dict = input_params_dict.copy()

        for key in keys:
            value = params_dict[key]
            setattr(self, key, value)

            # NB fails if input_params_dict missing required key.
            params_dict.pop(key)

        if params_dict:
            logger.warning(f"Skipping additional params in provided dict={params_dict}")

        self.cna_states = np.array(self.cna_states)
        self.num_states = len(self.cna_states)
        self.__verify()
