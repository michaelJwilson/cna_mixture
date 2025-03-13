import numpy as np
from numpy import random


class CNA_mixture_params:
    """
    Data class for parameters required by CNA mixture model,
    with shared overdispersions.
    """
    def __init__(self, num_cna_states=3, tau=50.0, phi=2.0e-2, genome_coverage=1.):
        """
        Initialize an instance of the class with random values in the assumed bounds.
        """
        # NB normal state is treated independently
        self.num_states = 1 + num_cna_states

        self.genome_coverage = genome_coverage
        
        # NB BAF overdispersion.  Random between 25. and 55.
        self.overdisp_tau = tau

        # NB RDR overdispersion.  Random between 1e-2 and 4e-2
        self.overdisp_phi = phi

        # NB list of (baf, rdr) for k=4 states, without replacement.
        integers = np.random.choice(
            np.arange(3, 10), size=num_cna_states, replace=False
        )
        integers = np.sort(integers)

        self.normal_state = [1.0, 0.5]
        self.cna_states = [
            [1.0 * int_sample, 1.0 / int_sample] for int_sample in integers
        ]

        self.cna_states = np.array([self.normal_state, *self.cna_states])
        self.normal_state = np.array(self.normal_state)

        self.__verify()

    def __verify(self):
        assert isinstance(
            self.cna_states, np.ndarray
        ), f"cna_states attribute must be a numpy array. Found {type(self.cna_states)}"

    def __str__(self):
        return ",  ".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    @property
    def params(self):
        read_depths = self.genome_coverage * self.cna_states[:,0]

        return np.array([
            *read_depths.tolist(),
            self.overdisp_phi,
            *self.cna_states[:,1].tolist(),
            self.overdisp_tau
        ])
        
    
    def dict_update(self, input_params_dict):
        """
        Update an instance of CNA_mixture_params to the input key: value dict.
        """
        keys = self.__dict__.keys()
        params_dict = input_params_dict.copy()

        for key in keys:
            value = params_dict[key]
            setattr(self, key, value)

            # NB fails if input_params_dict missing required key.
            params_dict.pop(key)

        if params_dict:
            f"Skipping additional params in dict={params_dict}"

        self.cna_states = np.array(self.cna_states)
        self.num_states = len(self.cna_states)
        self.__verify()

    def rdr_baf_choice_update(self, rdr_baf, threshold=0.05):
        non_normal = rdr_baf[np.abs(rdr_baf[:, 0] - 1.0) > threshold]

        xx = np.arange(len(non_normal))
        idx = random.choice(xx, size=self.num_states - 1, replace=False)
        
        self.cna_states = np.vstack([self.normal_state, non_normal[idx]])
        self.cna_states = self.cna_states[self.cna_states[:, 0].argsort()]        
