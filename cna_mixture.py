import logging
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from scipy.stats import nbinom, betabinom, poisson
from scipy.special import gamma
from scipy.special import logsumexp as logsumexp
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture

np.random.seed(1234)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def simple_logsumexp(array):
    max_val = array.max()
    shifted_array = array.copy() - max_val

    # TODO test: array = -np.arange(100); assert logsumexp(array) == __logsumexp(array)
    return max_val + np.log(np.exp(shifted_array).sum())


def assign_closest(points, centers):
    assert len(points) > len(centers)

    tree = KDTree(centers)
    distances, idx = tree.query(points)

    return idx


def onehot_encode_states(state_array):
    """
    Given an array of categorical states, return the
    (# samples, # states) one-hot encoding.

    NB equivalent to the state posterior!
    """
    num_states = np.max(state_array).astype(int) + 1
    states = state_array.astype(int)

    return np.eye(num_states)[states]


def reparameterize_nbinom(means, overdisp):
    """
    Reparameterize negative binomial from per-state means
    and shared overdispersion to (num_successes, prob. of success).
    """
    # NB https://en.wikipedia.org/wiki/Negative_binomial_distribution.
    means = np.array(means)

    # NB [0.0, 1.0] by definition.
    ps = 1.0 / (1.0 + overdisp * means)

    # NB for overdisp << 1, r >> 1, Gamma(r) -> Stirling's / overflow.
    rs = np.ones_like(means) / overdisp

    return np.c_[rs, ps]


def reparameterize_beta_binom(input_bafs, overdispersion):
    """
    Given the array of BAFs for all states and a shared overdispersion,
    return the (# states, 2) array of [alpha, beta] for each state,
    where beta is associated to the BAF probability.
    """
    return np.array(
        [
            [
                (1.0 - baf) * overdispersion,
                baf * overdispersion,
            ]
            for baf in input_bafs
        ]
    )


def categorical_state_logprobs(lambdas, num_samples):
    """ """
    ls = lambdas.copy()
    norm = np.sum(ls)

    assert np.abs(norm - 1.0) < 1.0e-6, "Lambdas are not accurately normalized"

    ls = np.log(ls)

    return np.broadcast_to(ls, (num_samples, len(ls))).copy()


def beta_binom_state_logprobs(state_alpha_betas, ks, ns):
    """
    Evaluate log prob. under BetaBinom model.
    Returns (# sample, # state) array.

    TODO port from python
    """
    result = np.zeros((len(ks), len(state_alpha_betas)))

    for col, (alpha, beta) in enumerate(state_alpha_betas):
        for row, (k, n) in enumerate(zip(ks, ns)):
            result[row, col] = betabinom.logpmf(k, n, beta, alpha)

    return result


def poisson_state_logprobs(state_mus, ks):
    result = np.zeros((len(ks), len(state_mus)))

    for col, mu in enumerate(state_mus):
        for row, kk in enumerate(ks):
            result[row, col] = poisson.logpmf(kk, mu)

    return result


def nbinom_state_logprobs(state_rs_ps, ks):
    """
    Evaluate log prob. under NegativeBinom model.
    Return (# sample, # state) array.

    TODO port from python
    """
    result = np.zeros((len(ks), len(state_rs_ps)))

    # TODO Poisson limit for (phi * mu) << 1.
    for col, (rr, pp) in enumerate(state_rs_ps):
        for row, kk in enumerate(ks):
            result[row, col] = nbinom.logpmf(kk, rr, pp)

    return result


def normalize_ln_posteriors(ln_posteriors):
    num_samples, num_states = ln_posteriors.shape

    # NB natural logarithm by definition;
    norm = logsumexp(ln_posteriors, axis=1)
    norm = np.broadcast_to(norm.reshape(num_samples, 1), (num_samples, num_states))

    return ln_posteriors.copy() - norm


class CNA_mixture_params:
    """
    Data class for parameters required by CNA mixture model with
    shared overdispersions.
    """

    def __init__(self):
        """
        Initialize an instance of the class with random values in
        the assumed bounds.
        """
        # NB normal is treated independently
        self.num_cna_states = 3
        self.num_states = 1 + self.num_cna_states

        # NB BAF overdispersion.  Random between 25. and 55.
        self.overdisp_tau = 25.0 + 30.0 * np.random.rand()

        # NB RDR overdispersion.  Random between 1e-2 and 4e-2
        self.overdisp_phi = 1.0e-2

        # NB list of (baf, rdr) for k=4 states.
        integer_samples = np.random.choice(
            np.arange(2, 10), size=self.num_cna_states, replace=False
        )
        integer_samples = np.sort(integer_samples)

        self.normal_state = [1.0, 0.5]
        self.cna_states = [
            [1.0 * int_sample, 1.0 / int_sample] for int_sample in integer_samples
        ]

        self.cna_states = [self.normal_state] + self.cna_states

        self.cna_states = np.array(self.cna_states)
        self.normal_state = np.array(self.normal_state)

        self.lambdas = np.random.rand(self.num_states)
        self.lambdas /= np.sum(self.lambdas)

        self.__verify()

    def update(self, input_params_dict):
        """
        Update an instance of CNA_mixture_params to the input key: value dict.
        """
        keys = self.__dict__.keys()
        params_dict = input_params_dict.copy()

        for key in keys:
            value = params_dict[key]
            setattr(self, key, value)

            params_dict.pop(key)

        assert (
            not params_dict
        ), f"Input params dict must include all of {keys}. Found {input_params_dict.keys()}"

        self.cna_states = np.array(self.cna_states)
        self.num_states = len(self.cna_states)
        self.__verify()

    def __verify(self):
        assert isinstance(
            self.cna_states, np.ndarray
        ), f"cna_states attribute must be a numpy array. Found {type(self.cna_states)}"

        logging.info(f"Simulating CNA states:\n{self.cna_states}")

    def __str__(self):
        printable = [f"{key}: {value}" for key, value in self.__dict__.items()]
        return ",  ".join(printable)


class CNA_Sim:
    def __init__(self):
        self.num_segments = 10_000

        # NB normal coverage per segment, i.e. for RDR=1.
        self.min_snp_coverage, self.max_snp_coverage, self.normal_genome_coverage = (
            100,
            1_000,
            500,
        )

        self.assumed_cna_mixture_params = {
            "overdisp_tau": 45.0,
            "overdisp_phi": 1.0e-2,
            "cna_states": [
                [3.0, 0.33],
                [4.0, 0.25],
                [10.0, 0.1],
            ],
            "normal_state": [1.0, 0.5],
            "lambdas": np.array([0.31807498, 0.06162617, 0.52047995, 0.0998189]),
        }

        for key, value in self.assumed_cna_mixture_params.items():
            setattr(self, key, value)

        self.cna_states = [self.normal_state] + self.cna_states

        # logger.info(f"Simulating copy number states: {self.cna_states}.")

        self.cna_states = np.array(self.cna_states)
        self.normal_state = np.array(self.normal_state)
        self.num_states = len(self.cna_states)

    def realize(self):
        """
        Generate a realization (one seed only) for given configuration settings.
        """
        # NB SNP-covering reads per segment.
        self.snp_coverages = np.random.randint(
            self.min_snp_coverage, self.max_snp_coverage, self.num_segments
        )

        result = []

        # NB we loop over genomic segments, sampling a state and assigning appropriate
        #    emission values.
        for ii in range(self.num_segments):
            # NB Equal-probability for categorical states: {0, .., K-1}.
            state = np.random.randint(0, self.num_states)
            rdr, baf = self.cna_states[state]

            # NB overdisp_tau parameterizes the degree of deviations from the mean baf.
            alpha, beta = reparameterize_beta_binom([baf], self.overdisp_tau)[0]

            # NB assumes some slop in terms of deviates from mean baf.
            b_reads = betabinom.rvs(self.snp_coverages[ii], beta, alpha)
            baf = b_reads / self.snp_coverages[ii]

            # NB we expect for baf ~0.5, some baf estimate to NOT be the minor allele,
            #    i.e. to occur at a rate > 0.5;

            true_read_coverage = rdr * self.normal_genome_coverage

            # NB equivalent to r and prob. for a bernoulli trial of r.
            lost_reads, dropout_rate = reparameterize_nbinom(
                [true_read_coverage], self.overdisp_phi
            )[0]

            read_coverage = nbinom.rvs(lost_reads, dropout_rate, size=1)[0]

            # NB CNA state, obs. transcripts (NegBin), lost transcripts (NegBin), B-allele support transcripts, vis a vis A.
            result.append(
                [
                    state,
                    read_coverage,
                    true_read_coverage,  # NB not an observable, to be inferrred.
                    b_reads,
                    self.snp_coverages[ii],
                ]
            )

        self.data = np.array(result)

    def get_data_bykey(self, key):
        keys = {
            "state": 0,
            "read_coverage": 1,
            "true_read_coverage": 2,
            "b_reads": 3,
            "snp_coverage": 4,
        }

        col = keys[key]

        return self.data[:, col]

    def plot_rdr_baf(self, rdr, baf, state_posteriors=None, states=None, title=None):
        """
        NB state_posteriors may be an integer, corresponding to a decoded state, or
           the posterior probs. for up to four states, which are mapped to RGB +
           alpha transparency.
        """
        if state_posteriors is not None:
            if state_posteriors.ndim == 1:
                rgb = state_posteriors
                alpha = 0.25
                cmap = "viridis"

            else:
                assert state_posteriors.shape[1] == 4

                # NB assumed to be normal probability.
                alpha = 0.25 + 3.0 * (1.0 - state_posteriors[:, 0]) / 4.0
                rgb = state_posteriors[:, 1:4]
                cmap = None

        pl.axhline(0.5, c="k", lw=0.5)
        plt.scatter(rdr, baf, c=rgb, marker=".", lw=0.0, alpha=alpha, cmap=cmap)

        if states is not None:
            for rdr, baf in states:
                pl.scatter(
                    rdr, baf, marker="*", edgecolors="black", facecolors="white", s=45
                )

        pl.xlim(-0.05, 15.0)
        pl.ylim(-0.05, 1.05)

        pl.xlabel(r"$\mu_{\rm RDR}$")
        pl.ylabel(r"$p_{\rm BAF}$")

        if title is not None:
            pl.title(title)

        pl.show()

    def plot_realization(self):
        """
        BAF vs RDR for the assumed simulation.
        """
        true_states = self.get_data_bykey("state")

        baf = self.get_data_bykey("b_reads") / self.get_data_bykey("snp_coverage")
        rdr = self.get_data_bykey("read_coverage") / self.normal_genome_coverage

        self.plot_rdr_baf(
            rdr,
            baf,
            state_posteriors=true_states,
            states=self.cna_states,
            title="CNA realizations",
        )

    def fit_gaussian_mixture(
        self,
        num_samples=100_000,
        num_components=4,
        random_state=0,
        max_iter=1,
        covariance_type="diag",
    ):
        """
        See:  https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/src/calicost/utils_hmm.py#L163
        """
        baf = self.get_data_bykey("b_reads") / self.get_data_bykey("snp_coverage")
        rdr = self.get_data_bykey("read_coverage") / self.normal_genome_coverage

        X = np.c_[rdr, baf]

        # NB covariance_type = {diag, full}
        #
        #    see https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        gmm = GaussianMixture(
            n_components=num_components,
            random_state=random_state,
            max_iter=max_iter,
            covariance_type=covariance_type,
        ).fit(X)

        means = np.c_[gmm.means_[:, 0], gmm.means_[:, 1]]
        samples, decoded_states = gmm.sample(n_samples=num_samples)

        logger.info(f"Fit Gaussian mixture means:\n{means}")

        self.plot_rdr_baf(
            samples[:, 0],
            samples[:, 1],
            state_posteriors=decoded_states,
            states=means,
            title=r"Gaussian Mixture Model samples",
        )

    def fit_cna_mixture(self):
        """
        Fit CNA mixture model via Expectation Maximization.
        Assumes RDR + BAF are independent given CNA state.

        See:
            https://udlbook.github.io/cvbook/

            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        """
        rdr = self.get_data_bykey("read_coverage") / self.normal_genome_coverage
        baf = self.get_data_bykey("b_reads") / self.get_data_bykey("snp_coverage")

        points = np.c_[rdr, baf]

        # NB defines initial (BAF, RDR) for each of K states and shared overdispersions.
        init_mixture_params = CNA_mixture_params()

        # TODO kmeans++ like.
        decoded_states = assign_closest(points, init_mixture_params.cna_states)

        # NB categorical prior on state fractions
        _, state_counts = np.unique(decoded_states, return_counts=True)
        state_lambdas = state_counts / np.sum(state_counts)

        state_rs_ps = reparameterize_nbinom(
            self.normal_genome_coverage * init_mixture_params.cna_states[:, 0],
            init_mixture_params.overdisp_phi,
        )

        state_alpha_betas = reparameterize_beta_binom(
            init_mixture_params.cna_states[:, 1],
            init_mixture_params.overdisp_tau,
        )

        # NB one-hot encoding of decoded state == ln. posterior.
        # ln_state_posteriors = onehot_encode_states(decoded_states)

        ln_state_posteriors = categorical_state_logprobs(
            state_lambdas,
            self.num_segments,
        )

        ln_state_posteriors += beta_binom_state_logprobs(
            state_alpha_betas,
            self.get_data_bykey("b_reads"),
            self.get_data_bykey("snp_coverage"),
        )

        ln_state_posteriors += nbinom_state_logprobs(
            state_rs_ps, self.get_data_bykey("read_coverage")
        )

        ln_state_posteriors = normalize_ln_posteriors(ln_state_posteriors)
        state_posteriors = np.exp(ln_state_posteriors)

        self.plot_rdr_baf(
            rdr,
            baf,
            state_posteriors=state_posteriors,
            states=init_mixture_params.cna_states,
        )


if __name__ == "__main__":
    cna_sim = CNA_Sim()
    cna_sim.realize()

    # cna_sim.plot_realization()
    # cna_sim.fit_gaussian_mixture()

    cna_sim.fit_cna_mixture()
