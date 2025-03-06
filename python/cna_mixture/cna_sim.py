import logging
import numpy as np

from scipy.stats import nbinom, betabinom, poisson
from cna_mixture.beta_binomial import reparameterize_beta_binom
from cna_mixture.negative_binomial import reparameterize_nbinom


logger = logging.getLogger(__name__)


def get_sim_params():
    return {
        "num_segments": 10_000,
        "num_states": 4,
        "jump_rate": 0.1,
        "normal_state": np.array([1.0, 0.5]),
        "cna_states": np.array([
            [1.0, 0.5],
            [3.0, 0.33],
            [4.0, 0.25],
            [10.0, 0.1],
        ]),
        "overdisp_tau": 45.0,
        "overdisp_phi": 1.0e-2,
        "min_snp_coverage": 100,
        "max_snp_coverage": 1_000,
        "normal_genome_coverage": 500 # NB normal coverage per segment, i.e. for RDR=1.
    }


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


class CNA_sim:
    def __init__(self):
        super().__init__()

        self.transfer = CNA_transfer()
    
        for key, value in get_sim_params().items():
            setattr(self, key, value)

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
            state = np.random.choice(np.arange(self.num_states), size=1, p=transfer_probs)[
                0
            ]

            rdr, baf = self.cna_states[state]

            # NB overdisp_tau parameterizes the degree of deviations from the mean baf.
            alpha, beta = reparameterize_beta_binom([baf], self.overdisp_tau)[0]

            # NB simulate variation in realized BAF according to betabinom model.
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
        # self.realized_genome_coverage = np.sum(self.data[:,2]) / self.num_segments

        self.realized_genome_coverage = self.normal_genome_coverage

    # TODO independent rdr and baf properties.
    @property
    def rdr_baf(self):
        rdr = self.data["read_coverage"] / self.realized_genome_coverage
        baf = self.data["b_reads"] / self.data["snp_coverage"]

        return np.c_[rdr, baf]

    def plot_realization_true_flat(self):
        """
        BAF vs RDR for the assumed simulation.
        """
        true_states = self.data["state"]

        plot_rdr_baf_flat(
            "plots/truth_rdr_baf_flat.pdf",
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=np.log(onehot_encode_states(true_states)),
            states_bag=self.cna_states,
            title="CNA realizations - true states",
        )
