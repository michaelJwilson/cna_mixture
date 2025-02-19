import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from scipy.stats import nbinom, betabinom
from scipy.special import gamma
from scipy.special import logsumexp as __logsumexp
from sklearn.mixture import GaussianMixture

np.random.seed(1234)


def logsumexp(array):
    max_val = array.max()
    shifted_array = array.copy() - max_val

    # TODO test: array = -np.arange(100); assert logsumexp(array) == __logsumexp(array)
    return max_val + np.log(np.exp(shifted_array).sum())


def onehot_encode_states(state_array):
    """
    Given an array of categorical states, return the
    (# samples, # states) one-hot encoding.
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
    variances = means + overdisp * means**2

    # NB [0.0, 1.0] by definition.
    ps = means / variances
    ns = means * ps / (1.0 - ps)

    return np.c_[ns, ps]


def reparameterize_beta_binom(input_bafs, overdispersion):
    """
    Given the array of BAFs for all states and a shared overdispersion,
    return the (# states, 2) array of [beta, alpha] for each state.
    """
    return np.array(
        [
            [
                baf * overdispersion,
                (1.0 - baf) * overdispersion,
            ]
            for baf in input_bafs
        ]
    )


def beta_binom_state_logprobs(state_alpha_betas, ks, ns):
    """
    Evaluate log prob. under BetaBinom model.
    Returns (# sample, # state) array.

    TODO port from python
    """
    result = np.zeros((len(ks), len(state_alpha_betas)))

    for col, (beta, alpha) in enumerate(state_alpha_betas):
        for row, (k, n) in enumerate(zip(ks, ns)):
            result[row, col] = betabinom.logpmf(k, n, alpha, beta)

    return result


def nbinom_state_logprobs(state_rs_ps, ks):
    """
    Evaluate log prob. under NegativeBinom model.
    Return (# sample, # state) array.

    TODO port from python
    """
    result = np.zeros((len(ks), len(state_rs_ps)))

    for col, (rr, pp) in enumerate(state_rs_ps):
        for row, kk in enumerate(ks):
            result[row, col] = nbinom.logpmf(kk, rr, pp)

    return result


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
        # NB BAF overdispersion.  Random between 25. and 55.
        self.overdisp_tau = 25.0 + 30.0 * np.random.rand()

        # NB RDR overdispersion.  Random between 1e-2 and 4e-2
        self.overdisp_phi = 1.0e-2

        # NB list of (baf, rdr) for k=4 states.
        integer_samples = np.random.randint(1, 10, 4)

        self.normal_state = [0.5, 1.0]
        self.cna_states = [
            [1.0 / int_sample, 1.0 * int_sample] for int_sample in integer_samples
        ]

        self.cna_states = [self.normal_state] + self.cna_states
        self.cna_states = np.array(self.cna_states)
        self.normal_state = np.array(self.normal_state)

        self.num_states = len(self.cna_states)
        self.__verify()

    def update(self, input_params_dict):
        """
        Update an instance of the class to the input key: value dict.
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

    def __str__(self):
        printable = [f"{key}: {value}" for key, value in self.__dict__.items()]
        return ",  ".join(printable)


class CNA_Sim:
    def __init__(self):
        self.num_segments = 10_000
        self.min_coverage, self.max_coverage = 100, 1_000

        self.assumed_cna_mixture_params = {
            "overdisp_tau": 45.0,
            "overdisp_phi": 1.0e-2,
            "cna_states": [
                [0.1, 10.0],
                [0.25, 4.0],
                [0.33, 3.0],
            ],
            "normal_state": [0.5, 1.0],
        }

        for key, value in self.assumed_cna_mixture_params.items():
            setattr(self, key, value)

        self.cna_states = [self.normal_state] + self.cna_states

        self.cna_states = np.array(self.cna_states)
        self.normal_state = np.array(self.normal_state)
        self.num_states = len(self.cna_states)

    def realize(self):
        """
        Generate a realization (one seed only) for given configuration settings.
        """
        # NB SNP-covering reads per segment.
        self.snp_coverages = np.random.randint(
            self.min_coverage, self.max_coverage, self.num_segments
        )

        # NB normal coverage per segment, i.e. for RDR=1.
        self.normal_coverages = self.snp_coverages.copy() + np.random.randint(
            self.min_coverage, self.max_coverage, self.num_segments
        )

        result = []

        # NB we loop over genomic segments, sampling a state and assigning appropriate
        #    emission values.
        for ii in range(self.num_segments):
            # NB Equal-probability for categorical states: {0, .., K-1}.
            state = np.random.randint(0, self.num_states)
            baf, rdr = self.cna_states[state]

            # NB overdisp_tau parameterizes the degree of deviations from the mean baf.
            beta, alpha = reparameterize_beta_binom([baf], self.overdisp_tau)[0]

            # NB assumes some slop in terms of deviates from mean baf.
            b_reads = betabinom.rvs(self.snp_coverages[ii], alpha, beta)

            true_read_coverage = rdr * self.normal_coverages[ii]
            lost_reads, dropout_rate = reparameterize_nbinom(
                [true_read_coverage], self.overdisp_phi
            )[0]

            read_coverage = nbinom.rvs(lost_reads, dropout_rate, size=1)[0]

            # NB CNA state, obs. transcripts (NegBin), lost transcripts (NegBin), B-allele support transcripts, vis a vis A.
            result.append(
                [
                    state,
                    read_coverage,
                    true_read_coverage,
                    b_reads,
                    self.snp_coverages[ii],
                    self.normal_coverages[ii],
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
            "normal_coverage": 5,
        }

        col = keys[key]

        return self.data[:, col]

    def plot_realization(self):
        realized_states = self.get_data_bykey("state")

        baf = self.get_data_bykey("b_reads") / self.get_data_bykey("snp_coverage")
        rdr = self.get_data_bykey("read_coverage") / self.get_data_bykey(
            "normal_coverage"
        )

        plt.scatter(rdr, baf, c=realized_states, marker=".", lw=0.0, alpha=0.25)

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
        baf = self.get_data_bykey("b_reads") / self.get_data_bykey("snp_coverage")
        rdr = self.get_data_bykey("read_coverage") / self.get_data_bykey(
            "normal_coverage"
        )

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

    def fit_cna_mixture(self):
        """
        Fit CNA mixture model via Expectation Maximization.  Assumes RDR + BAF are independent
        given CNA state.

        See:
            https://udlbook.github.io/cvbook/

            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        """
        # TODO
        # initialize random CNA state parameters + overdispersions
        #
        # while not converged:
        #     assign state posteriors given current parameters
        #     update paramaters given state posteriors.

        # NB defines initial (BAF, RDR) for each of K states and shared overdispersions.
        init_mixture_params = CNA_mixture_params()

        state_alpha_betas = reparameterize_beta_binom(
            init_mixture_params.cna_states[:, 0],
            init_mixture_params.overdisp_tau,
        )

        # TODO first column values are all the same??  overdisp_phi << 1?
        state_rs_ps = reparameterize_nbinom(
            init_mixture_params.cna_states[:, 1], init_mixture_params.overdisp_phi
        )

        print(state_rs_ps)

        # print(state_alpha_betas)
        # print(state_rs_ps)

        # NB initial responsibilites are categorial prior on probability of each state,
        #    i.e. no emission probabilities.
        init_responsibilities = np.random.rand(init_mixture_params.num_states)
        init_responsibilities /= np.sum(init_responsibilities)

        init_ln_state_posteriors = np.log(init_responsibilities)

        # NB - fed with num b reads and total snp covering reads.
        #    - broadcast (# state, 1) to (# samples, # states)
        ln_state_posteriors = beta_binom_state_logprobs(
            state_alpha_betas,
            self.get_data_bykey("b_reads"),
            self.get_data_bykey("snp_coverage"),
        )

        ln_state_posteriors += nbinom_state_logprobs(
            state_rs_ps, self.get_data_bykey("read_coverage")
        )

        ln_state_posteriors += init_ln_state_posteriors

        """
        # NB increment with ln BAF prob. and ln RDR prob., assuming independent given state.
        #
        # NB ln_baf_prob == (n_segment x n_state)
        
        # normalize 
        
        print(ln_state_posteriors)
        """


if __name__ == "__main__":
    cna_sim = CNA_Sim()
    cna_sim.realize()

    # cna_sim.plot_realization()
    # cna_sim.fit_gaussian_mixture()

    # cna_sim.fit_cna_mixture()
