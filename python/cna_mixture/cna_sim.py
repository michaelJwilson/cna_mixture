import json
import logging
from pathlib import Path

import numpy as np
from rich.pretty import pprint
from scipy.stats import betabinom, nbinom

from cna_mixture.cna_emission import reparameterize_beta_binom, reparameterize_nbinom
from cna_mixture.encoding import onehot_encode_states
from cna_mixture.plotting import plot_rdr_baf_flat, plot_rdr_baf_genome
from cna_mixture.transfer import CNA_transfer

logger = logging.getLogger(__name__)


def get_sim_params():
    return {
        "num_segments": 10_000,
        "num_states": 4,
        "jump_rate": 0.1,
        "normal_state": np.array([1.0, 0.5]),
        "cna_states": np.array(
            [
                [1.0, 0.5],
                [3.0, 0.33],
                [4.0, 0.25],
                [10.0, 0.1],
            ]
        ),
        "overdisp_tau": 45.0,
        "overdisp_phi": 1.0e-2,
        "min_snp_coverage": 100,
        "max_snp_coverage": 1_000,
        "normal_genome_coverage": 500,  # NB normal coverage per segment, i.e. for RDR=1.
    }


def get_sim_output_dtype():
    return [
        ("state", np.float64),
        ("read_coverage", np.float64),
        ("true_read_coverage", np.float64),
        ("b_reads", np.float64),
        ("snp_coverage", np.float64),
    ]


class CNA_sim:
    """
    A 1D genome simulation for a CNA markov model with NB/BB emission models.
    """
    def __init__(self, sim_id=0, params=None, data=None, seed=314):
        super().__init__()

        self.sim_id = sim_id
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.params = params if params is not None else get_sim_params()
        
        for key, value in self.params.items():
            setattr(self, key, value)

        self.transfer = CNA_transfer(self.jump_rate, self.num_states)
            
        if data is None:
            # NB guard against inconsistent data/params.
            assert (
                params is None
            ), "Parameters must not be provided when generating new data"

            self.data = self.realize_data()
        else:
            assert (
                params is not None
            ), "Parameters are required when loading pre-generated data."

            self.data = data

        # NB if rdr=1 always, equates == self.num_segments * self.normal_genome_coverage
        # TODO? biases RDR estimates, particularly if many CNAs.
        #
        # self.genome_coverage = np.sum(self.data[:,2]) / self.num_segments

        self.genome_coverage = self.normal_genome_coverage

    def print(self):
        print(f"\nCNA_Sim({self.sim_id})=")
        pprint(self.params)
        print(f"with data=\n{self.data}")

    def realize_data(self):
        """
        Generate a realization (one seed only) for given configuration settings.
        """
        logger.info(f"Simulating copy number states:\n{self.cna_states}.")

        # NB SNP-covering reads per segment.
        snp_coverages = self.rng.integers(
            self.min_snp_coverage, self.max_snp_coverage, self.num_segments
        )

        result = []

        # NB Equal-probability for categorical states: {0, .., K-1}.
        state = self.rng.integers(0, self.num_states)

        # NB we loop over genomic segments, sampling a state and assigning appropriate
        #    emission values.
        for ii in range(self.num_segments):
            transfer_probs = self.transfer.transfer_matrix[state]
            state = self.rng.choice(
                np.arange(self.num_states), size=1, p=transfer_probs
            )[0]

            rdr, baf = self.cna_states[state]

            # NB overdisp_tau parameterizes the degree of deviations from the mean baf.
            alpha, beta = reparameterize_beta_binom([baf], self.overdisp_tau)[0]

            # NB simulate variation in realized BAF according to betabinom model;
            #    b_reads have an expected BAF, as encoded by beta.
            b_reads = betabinom.rvs(snp_coverages[ii], beta, alpha, random_state=self.rng)

            # NB we expect for baf ~0.5, some baf estimate to NOT be the minor allele,
            #    i.e. to occur at a rate > 0.5;
            baf = b_reads / snp_coverages[ii]

            # NB stochastic, given rdr derived from state sampling.
            true_read_coverage = rdr * self.normal_genome_coverage

            # NB equivalent to r and prob. for a bernoulli trial of r.
            lost_reads, dropout_rate = reparameterize_nbinom(
                [true_read_coverage], self.overdisp_phi
            )[0]

            read_coverage = nbinom.rvs(lost_reads, dropout_rate, size=1, random_state=self.rng)[0]

            # NB CNA state, obs. transcripts (NegBin), lost transcripts (NegBin), B-allele support transcripts, vis a vis A.
            result.append(
                (
                    state,
                    read_coverage,
                    true_read_coverage,  # NB not an observable, to be inferrred.
                    b_reads,
                    snp_coverages[ii],
                )
            )

        return np.array(result, dtype=get_sim_output_dtype())

    def save(self, output_dir):
        sim_params = self.params.copy()
        sim_params["genome_coverage"] = self.genome_coverage
        sim_params["seed"] = self.seed

        sim_params = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in sim_params.items()
        }

        with Path(f"{output_dir}/cna_sim_parameters.json").open("w") as ff:
            json.dump(sim_params, ff, indent=4)

        Path(f"{output_dir}/cna_sim_{self.sim_id}").mkdir(exist_ok=True, parents=True)
            
        np.savetxt(
            f"{output_dir}/cna_sim_{self.sim_id}/cna_sim_data_{self.sim_id}.txt",
            self.data,
            delimiter="\t",
            header=",".join(self.data.dtype.names),
            fmt="%.6f",
        )

        logger.info(f"Successfully saved sim. {self.sim_id} output to {output_dir}")

    @classmethod
    def load(cls, output_dir, sim_id):
        # TODO guard against missing/corrupted file.
        params_path = f"{output_dir}/cna_sim_parameters.json"
        
        with Path.open(params_path) as ff:
            params = json.load(ff)

        logger.info(f"Loading simulation parameters @ {params_path}")
            
        data_path = f"{output_dir}/cna_sim_{sim_id}/cna_sim_data_{sim_id}.txt"            
        data = np.loadtxt(data_path, dtype=np.float64)

        logger.info(f"Loading simulation data @ {data_path}")

        # TODO UGH comprehension is slow
        data = np.array([tuple(row) for row in data], dtype=get_sim_output_dtype())
        
        cna_sim = CNA_sim(sim_id=sim_id, params=params, data=data)
        cna_sim.print()

        return cna_sim

    @property
    def rdr(self):
        return self.data["read_coverage"] / self.genome_coverage

    @property
    def baf(self):
        return self.data["b_reads"] / self.data["snp_coverage"]

    @property
    def rdr_baf(self):
        return np.c_[self.rdr, self.baf]

    def plot_realization_true_flat(self, fpath):
        """
        BAF vs RDR for the assumed simulation.
        """
        plot_rdr_baf_flat(
            fpath,
            self.rdr,
            self.baf,
            ln_state_posteriors=np.log(onehot_encode_states(self.data["state"])),
            states_bag=self.cna_states,
            title="CNA realizations - true states",
        )

    def plot_realization_true_genome(self, fpath):
        plot_rdr_baf_genome(
            fpath,
            self.rdr,
            self.baf,
            ln_state_posteriors=np.log(onehot_encode_states(self.data["state"])),
            states_bag=self.cna_states,
            title="CNA realizations - true states",
        )
