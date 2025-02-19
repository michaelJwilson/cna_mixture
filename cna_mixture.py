import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from scipy.stats import nbinom, betabinom
from sklearn.mixture import GaussianMixture

np.random.seed(1234)


def reparameterize_nbinom(mean, overdisp):
    """
    Reparameterize negative binomial from (mean, overdispersion)
    to (num_successes, prob. of success).
    """
    # NB https://en.wikipedia.org/wiki/Negative_binomial_distribution.
    var = mean + overdisp * mean**2

    p = mean / var
    n = mean * p / (1.0 - p)

    return (n, p)


class CNA_Sim:
    def __init__(self):
        # NB BAF overdispersion.
        self.overdisp_tau = 45.0

        # NB RDR overdispersion;
        self.overdisp_phi = 1.0e-2

        # NB list of (baf, rdr) for k=4 states.
        self.sim_cna_states = [
            [0.1, 10.0],
            [0.25, 4.0],
            [0.33, 3.0],
            [0.5, 1.0],  # normal
        ]

        # NB SNP-covering reads per segment.
        self.num_segments = 10_000
        self.min_snp_coverage, self.max_snp_coverage = 100, 1_000

        self.snp_coverages = np.random.randint(
            self.min_snp_coverage, self.max_snp_coverage, self.num_segments
        )

        # NB DEBUG
        self.total_coverages = 3 * self.snp_coverages.copy()

    def realize(self):
        result = []

        for ii in range(self.num_segments):
            # NB Equal-probability for categorical states: {0, .., K-1}.
            kk = np.random.randint(0, len(self.sim_cna_states))

            baf, rdr = self.sim_cna_states[kk]

            # NB overdisp_tau parameterizes the degree of deviations from the mean baf.
            beta, alpha = baf * self.overdisp_tau, (1.0 - baf) * self.overdisp_tau

            sim_b_reads = betabinom.rvs(self.snp_coverages[ii], alpha, beta)
            sim_a_reads = self.snp_coverages[ii] - sim_b_reads

            sim_baf = sim_b_reads / self.snp_coverages[ii]

            sim_lost_reads, dropout_rate = reparameterize_nbinom(
                rdr * self.total_coverages[ii], self.overdisp_phi
            )
            sim_retained_reads = nbinom.rvs(sim_lost_reads, dropout_rate, size=1)[0]

            # NB CNA state, obs. transcripts (NegBin), lost transcripts (NegBin), B-allele support transcripts, vis a vis A.
            result.append(
                [kk, sim_retained_reads, sim_lost_reads, sim_b_reads, sim_a_reads]
            )

        self.data = np.array(result)

    def plot_realization(self):
        realized_states = self.data[:, 0]

        sim_rdr = self.data[:, 1] / self.total_coverages
        sim_baf = self.data[:, 3] / (self.data[:, 3] + self.data[:, 4])

        plt.scatter(sim_rdr, sim_baf, c=realized_states, marker=".", lw=0.0, alpha=0.25)

        pl.axhline(1.0, c="k", lw=0.75)
        
        pl.xlim(-0.05, 15.0)
        pl.ylim(-0.05, 1.05)

        pl.xlabel(r"$\mu_{\rm RDR}$")
        pl.ylabel(r"$p_{\rm BAF}$")

        pl.title(r"CNA realization")

        pl.show()

    def fit_gaussian_mixture(
        self,
        num_samples=100_000,
        num_components=4,
        random_state=0,
        max_iter=1,
        covariance_type="diag",
    ):
        """
        see:  https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/src/calicost/utils_hmm.py#L163
        """
        sim_rdr = self.data[:, 1] / self.total_coverages
        sim_baf = self.data[:, 3] / (self.data[:, 3] + self.data[:, 4])

        X = np.c_[sim_rdr, sim_baf]

        # NB covariance_type = {diag, full}
        #
        #    see https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        gmm = GaussianMixture(
            n_components=num_components,
            random_state=random_state,
            max_iter=max_iter,
            covariance_type=covariance_type,
        ).fit(X)

        obs_data, states = gmm.sample(n_samples=num_samples)

        plt.scatter(
            obs_data[:, 0], obs_data[:, 1], c=states, marker=".", lw=0.0, alpha=0.25
        )

        pl.axhline(1.0, c="k", lw=0.75)
        
        pl.xlim(-0.05, 15.0)
        pl.ylim(-0.05, 1.05)

        pl.title(r"Gaussian Mixture Model samples")

        pl.xlabel(r"$\mu_{\rm RDR}$")
        pl.ylabel(r"$p_{\rm BAF}$")

        pl.show()

    def eval_log_prob():
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        """

        raise NotImplementedError()

        # NB assumes BAF and RDR are independent.
        # logpmf = betabinom.logpmf(k, n, a, b, loc=0) + nbinom.logpmf()


if __name__ == "__main__":
    cna_sim = CNA_Sim()
    cna_sim.realize()

    print(cna_sim.data[:, :15])

    cna_sim.plot_realization()
    cna_sim.fit_gaussian_mixture()
