import logging

import numpy as np
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


class CNA_sim:
    """
    A 1D genome simulation for a CNA markov model with NB/BB emission models.
    """
    def __init__(self):
        super().__init__()

        for key, value in get_sim_params().items():
            setattr(self, key, value)

        self.transfer = CNA_transfer(self.jump_rate, self.num_states)
        self.realize()

    def realize(self):
        """
        Generate a realization (one seed only) for given configuration settings.
        """
        logger.info(f"Simulating copy number states:\n{self.cna_states}.")

        # NB SNP-covering reads per segment.
        self.snp_coverages = np.random.randint(
            self.min_snp_coverage, self.max_snp_coverage, self.num_segments
        )

        result = []

        # NB Equal-probability for categorical states: {0, .., K-1}.
        state = np.random.randint(0, self.num_states)

        # NB we loop over genomic segments, sampling a state and assigning appropriate
        #    emission values.
        for ii in range(self.num_segments):
            transfer_probs = self.transfer.transfer_matrix[state]
            state = np.random.choice(
                np.arange(self.num_states), size=1, p=transfer_probs
            )[0]

            rdr, baf = self.cna_states[state]

            # NB overdisp_tau parameterizes the degree of deviations from the mean baf.
            alpha, beta = reparameterize_beta_binom([baf], self.overdisp_tau)[0]

            # NB simulate variation in realized BAF according to betabinom model;
            #    b_reads have an expected BAF, as encoded by beta.
            b_reads = betabinom.rvs(self.snp_coverages[ii], beta, alpha)

            # NB we expect for baf ~0.5, some baf estimate to NOT be the minor allele,
            #    i.e. to occur at a rate > 0.5;
            baf = b_reads / self.snp_coverages[ii]

            # NB stochastic, given rdr derived from state sampling.
            true_read_coverage = rdr * self.normal_genome_coverage

            # NB equivalent to r and prob. for a bernoulli trial of r.
            lost_reads, dropout_rate = reparameterize_nbinom(
                [true_read_coverage], self.overdisp_phi
            )[0]

            read_coverage = nbinom.rvs(lost_reads, dropout_rate, size=1)[0]

            # NB CNA state, obs. transcripts (NegBin), lost transcripts (NegBin), B-allele support transcripts, vis a vis A.
            result.append(
                (
                    state,
                    read_coverage,
                    true_read_coverage,  # NB not an observable, to be inferrred.
                    b_reads,
                    self.snp_coverages[ii],
                )
            )

        dtype = [
            ("state", np.float64),
            ("read_coverage", np.float64),
            ("true_read_coverage", np.float64),
            ("b_reads", np.float64),
            ("snp_coverage", np.float64),
        ]

        self.data = np.array(result, dtype=dtype)

        # NB if rdr=1 always, equates == self.num_segments * self.normal_genome_coverage
        # TODO? biases RDR estimates, particularly if many CNAs.
        #
        # self.genome_coverage = np.sum(self.data[:,2]) / self.num_segments

        self.genome_coverage = self.normal_genome_coverage

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
