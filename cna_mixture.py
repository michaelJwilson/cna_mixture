import logging
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from scipy.stats import nbinom, betabinom, poisson
from scipy.special import gamma
from scipy.special import logsumexp as logsumexp
from scipy.spatial import KDTree
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture

np.random.seed(1234)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def tophat_smooth(data, window_size):
    kernel = np.ones(window_size) / window_size

    # Apply convolution with 'same' mode to keep the output size the same as the input
    smoothed_data = np.convolve(data, kernel, mode="same")

    return smoothed_data


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

    NB equivalent to a state posterior!
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


def poisson_state_logprobs(state_mus, ks):
    result = np.zeros((len(ks), len(state_mus)))

    for col, mu in enumerate(state_mus):
        for row, kk in enumerate(ks):
            result[row, col] = poisson.logpmf(kk, mu)

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
        self.overdisp_tau = 45.0

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

    def __str__(self):
        return ",  ".join([f"{key}: {value}" for key, value in self.__dict__.items()])


class CNA_Sim:
    def __init__(self):
        self.num_segments = 10_000

        # TODO numerical precision on ps -> tie jump_rate to # states?
        self.jump_rate = 1.0e-1

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
        }

        for key, value in self.assumed_cna_mixture_params.items():
            setattr(self, key, value)

        self.cna_states = [self.normal_state] + self.cna_states

        self.cna_states = np.array(self.cna_states)
        self.normal_state = np.array(self.normal_state)
        self.num_states = len(self.cna_states)

        self.jump_rate_per_state = 1.0e-1 / (self.num_states - 1.0)
        self.transfer = self.jump_rate_per_state * np.ones(
            shape=(self.num_states, self.num_states)
        )
        self.transfer -= self.jump_rate_per_state * np.eye(self.num_states)
        self.transfer += (1.0 - self.jump_rate) * np.eye(self.num_states)

        self.realize()

    def realize(self):
        """
        Generate a realization (one seed only) for given configuration settings.
        """
        logger.info(f"Simulating copy number states: {self.cna_states}.")

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
            state_probs = self.transfer[state]
            state = np.random.choice(np.arange(self.num_states), size=1, p=state_probs)[
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
                [
                    state,
                    read_coverage,
                    true_read_coverage,  # NB not an observable, to be inferrred.
                    b_reads,
                    self.snp_coverages[ii],
                ]
            )

        self.data = np.array(result)

        # NB if rdr=1 always, equates == self.num_segments * self.normal_genome_coverage
        # TODO? biases RDR estimates, particularly if many CNAs.
        #
        # self.realized_genome_coverage = np.sum(self.data[:,2]) / self.num_segments

        self.realized_genome_coverage = self.normal_genome_coverage

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

    @property
    def rdr_baf(self):
        rdr = self.get_data_bykey("read_coverage") / self.realized_genome_coverage
        baf = self.get_data_bykey("b_reads") / self.get_data_bykey("snp_coverage")

        return np.c_[rdr, baf]

    def plot_rdr_baf_flat(
        self, rdr, baf, ln_state_posteriors=None, states_bag=None, title=None
    ):
        """
        NB state_posteriors may be an integer, corresponding to a decoded state, or
           the posterior probs. for up to four states, which are mapped to RGB +
           alpha transparency.
        """
        if ln_state_posteriors is not None:
            if ln_state_posteriors.ndim == 1:
                rgb = np.exp(ln_state_posteriors)
                alpha = 0.25
                cmap = "viridis"

            else:
                assert ln_state_posteriors.shape[1] == 4

                # NB assumed to be normal probability.
                # alpha = 0.25 + 3.0 * (1.0 - state_posteriors[:, 0]) / 4.0
                alpha = 0.25

                rgb = np.exp(ln_state_posteriors[:, 1:4])
                cmap = None

        pl.axhline(0.5, c="k", lw=0.5)
        plt.scatter(rdr, baf, c=rgb, marker=".", lw=0.0, alpha=alpha, cmap=cmap)

        if states_bag is not None:
            for rdr, baf in states_bag:
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

    def plot_realization_flat(self):
        """
        BAF vs RDR for the assumed simulation.
        """
        true_states = self.get_data_bykey("state")

        self.plot_rdr_baf_flat(
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=np.log(onehot_encode_states(true_states)),
            states_bag=self.cna_states,
            title="CNA realizations - true states",
        )

    def plot_realization_genome(
        self, ln_state_posteriors=None, states_bag=None, title=None
    ):
        bases = np.arange(self.num_segments)

        figsize = (15, 10)
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)

        rdr, baf = self.rdr_baf[:, 0], self.rdr_baf[:, 1]

        smooth_rdr = tophat_smooth(rdr, window_size=25)
        smooth_baf = tophat_smooth(baf, window_size=25)

        axes[0].plot(bases, rdr)
        axes[0].plot(bases, smooth_rdr)

        axes[1].plot(bases, baf)
        axes[1].plot(bases, smooth_baf)

        axes[0].set_ylabel(r"read depth ratio")

        axes[1].set_ylabel(r"$b$-allele frequency")
        axes[1].set_xlabel("intervals")

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
        See:  https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/src/calicost/utils_hmm.py#L163
        """
        # NB covariance_type = {diag, full}
        #
        #    see https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        gmm = GaussianMixture(
            n_components=num_components,
            random_state=random_state,
            max_iter=max_iter,
            covariance_type=covariance_type,
        ).fit(self.rdr_baf)

        means = np.c_[gmm.means_[:, 0], gmm.means_[:, 1]]
        samples, decoded_states = gmm.sample(n_samples=num_samples)

        logger.info(f"Fit Gaussian mixture means:\n{means}")

        self.plot_rdr_baf_flat(
            samples[:, 0],
            samples[:, 1],
            ln_state_posteriors=np.log(onehot_encode_states(decoded_states)),
            states_bag=means,
            title=r"Best-fit Gaussian Mixture Model samples",
        )

    def unpack_cna_mixture_params(self, params):
        """
        Given a cost parameter vector, unpack into named cna mixture
        parameters.
        """
        num_states = self.num_states

        # NB read_depths + overdispersion + bafs + overdispersion
        assert len(params) == num_states + 1 + num_states + 1

        state_read_depths = params[:num_states]
        rdr_overdispersion = params[num_states]

        bafs = params[num_states + 1 : 2 * num_states + 1]
        baf_overdispersion = params[2 * num_states + 1]

        return state_read_depths, rdr_overdispersion, bafs, baf_overdispersion

    def cna_mixture_categorical_update(self, ln_lambdas):
        """
        Broadcast per-state categorical priors to equivalent (samples x state)
        array.
        """
        ln_norm = logsumexp(ln_lambdas)

        # NB ensure normalized.
        return np.broadcast_to(
            ln_lambdas - ln_norm, (self.num_segments, len(ln_lambdas))
        ).copy()

    def cna_mixture_betabinom_update(self, params):
        """
        Evaluate log prob. under BetaBinom model.
        Returns (# sample, # state) array.
        """
        _, _, bafs, baf_overdispersion = self.unpack_cna_mixture_params(params)

        state_alpha_betas = reparameterize_beta_binom(
            bafs,
            baf_overdispersion,
        )

        ks, ns = self.get_data_bykey("b_reads"), self.get_data_bykey("snp_coverage")
        result = np.zeros((len(ks), len(state_alpha_betas)))

        # TODO port from python.  broadcast gammas.
        for col, (alpha, beta) in enumerate(state_alpha_betas):
            for row, (k, n) in enumerate(zip(ks, ns)):
                result[row, col] = betabinom.logpmf(k, n, beta, alpha)

        return result, state_alpha_betas

    def cna_mixture_nbinom_update(self, params):
        """
        Evaluate log prob. under NegativeBinom model.
        Return (# sample, # state) array.
        """
        state_read_depths, rdr_overdispersion, _, _ = self.unpack_cna_mixture_params(
            params
        )

        # TODO does a non-linear transform in the cost trip the optimizer?
        state_rs_ps = reparameterize_nbinom(
            state_read_depths,
            rdr_overdispersion,
        )

        ks = self.get_data_bykey("read_coverage")
        result = np.zeros((len(ks), len(state_rs_ps)))

        # TODO Poisson limit for (phi * mu) << 1.
        # TODO port from python.  broadcast gammas.
        for col, (rr, pp) in enumerate(state_rs_ps):
            for row, kk in enumerate(ks):
                result[row, col] = nbinom.logpmf(kk, rr, pp)

        return result, state_rs_ps

    def cna_mixture_ln_state_posterior_update(self, params, ln_lambdas):
        """
        Calculate *un-normalized* state posteriors based on current parameter + lambda
        settings.
        """
        ln_state_posterior_categorical = self.cna_mixture_categorical_update(ln_lambdas)
        ln_state_posterior_betabinom, _ = self.cna_mixture_betabinom_update(params)
        ln_state_posterior_nbinom, _ = self.cna_mixture_nbinom_update(params)

        # NB WARNING state posteriors are *not* normalized here, i.e. P(xi, hi), as required by EM cost.
        return (
            ln_state_posterior_categorical
            + ln_state_posterior_betabinom
            + ln_state_posterior_nbinom
        )

    def estep(self, params, ln_lambdas):
        """
        Calculate normalized state posteriors based on current parameter + lambda
        settings.
        """
        return normalize_ln_posteriors(
            self.cna_mixture_ln_state_posterior_update(params, ln_lambdas)
        )

    def cna_mixture_ln_lambdas_update(self, ln_state_posteriors):
        """ """
        return logsumexp(ln_state_posteriors, axis=0) - logsumexp(
            ln_state_posteriors
        )

    def cna_mixture_em_cost(self, params, ln_lambdas, approx_ln_state_posteriors=None, verbose=False):
        """
        if state_posteriors is provided, resulting EM-cost is a lower bound to the log likelihood at
        the current params values and the assumed state_posteriors.
        """
        # NB WARNING state posteriors are *not* normalized here, i.e. P(xi, hi) as required by EM cost.
        ln_state_posteriors_nonorm = self.cna_mixture_ln_state_posterior_update(params, ln_lambdas)

        if approx_ln_state_posteriors is None:
            # NB set ln_state_posteriors based on current parameters.
            ln_state_posteriors = normalize_ln_posteriors(ln_state_posteriors_nonorm)
        else:
            # NB utilize estimate of ln_state_posteriors, e.g. based on last parameter set rather than
            #    current.
            ln_state_posteriors = normalize_ln_posteriors(approx_ln_state_posteriors)

        # NB responsibilites rik, where i is the sample and k is the state.                                                                                                                                                    
        state_posteriors = np.exp(ln_state_posteriors)
             
        # NB this is *not* state-posterior weighted log-likelihood. 
        em_cost = state_posteriors * ln_state_posteriors_nonorm

        # NB sum over samples and states.  Maximization -> minimization.
        em_cost = -em_cost.sum()

        if verbose:
            state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
                self.unpack_cna_mixture_params(params)
            )

            msg = f"Minimizing cost with SLSQP with initial value: {em_cost} for:\n"
            msg += f"lambdas={np.exp(ln_lambdas)}\nread_depths={state_read_depths}\nread_depth_overdispersion={rdr_overdispersion}\n"
            msg += f"bafs={bafs}\nbaf_overdispersion={baf_overdispersion}"

            logger.info(msg)
            
        return em_cost

    def initialize_ln_lambdas(self, init_mixture_params):
        # TODO kmeans++ like.
        decoded_states = assign_closest(self.rdr_baf, init_mixture_params.cna_states)

        # NB categorical prior on state fractions
        _, counts = np.unique(decoded_states, return_counts=True)
        initial_ln_lambdas = np.log(counts) - np.log(np.sum(counts))

        return initial_ln_lambdas

    def fit_cna_mixture(self):
        """
        Fit CNA mixture model via Expectation Maximization.
        Assumes RDR + BAF are independent given CNA state.

        See:
            https://udlbook.github.io/cvbook/

            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        """
        # NB defines initial (BAF, RDR) for each of K states and shared overdispersions.
        init_mixture_params = CNA_mixture_params()
        initial_ln_lambdas = self.initialize_ln_lambdas(init_mixture_params)
        
        logging.info(f"Initializing CNA states:\n{init_mixture_params.cna_states}\n")

        # NB self.realized_genome_coverage == normal_coverage currently.
        initial_state_read_depths = (
            self.realized_genome_coverage * init_mixture_params.cna_states[:, 0]
        )
        initial_bafs = init_mixture_params.cna_states[:, 1]

        # NB e.g. [0.2443, 0.3857, 0.1247, 0.2453, ... 500.0, 1500.0, 2500.0, 3500.0, 0.01, ... 0.5, 0.3333333333333333, 0.2, 0.14285714285714285, 47.075625069001084]
        initial_params = (
            initial_state_read_depths.tolist()
            + [init_mixture_params.overdisp_phi]
            + initial_bafs.tolist()
            + [init_mixture_params.overdisp_tau]
        )

        initial_cost = self.cna_mixture_em_cost(initial_params, initial_ln_lambdas, verbose=True)

        ln_state_posteriors = self.estep(initial_params, initial_ln_lambdas)
        """
        self.plot_rdr_baf_flat(
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=ln_state_posteriors,
            states_bag=init_mixture_params.cna_states,
            title="Initial state posteriors (based on closest state lambdas)."
        )
        """
        
        # NB equality constaints to be zero.
        # TODO regularizer for state overlap?
        constraints = [
            # NB sum of RDRs should explain realized genome-wide coverage.
            {
                "type": "eq",
                "fun": lambda x: np.sum(x[: self.num_states])
                - self.realized_genome_coverage,
            },
        ]

        # NB all parameters are constained to be positive. bafs. max of unity.
        bounds = [(1.0e-6, None) for _ in range(self.num_states)]  # exp_read_depths
        bounds += [(1.0e-6, None)]  # RDR overdispersion
        bounds += [
            (1.0e-6, 1.0) for _ in range(self.num_states)
        ]  # bafs - not limited to 0.5
        bounds += [(1.0e-6, None)]  # baf overdispersion
        bounds = tuple(bounds)

        # NB https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
        res = minimize(
            self.cna_mixture_em_cost,
            initial_params,
            args=(initial_ln_lambdas),
            method="nelder-mead",
            bounds=bounds,
            constraints=None,
            options={"disp": True, "maxiter": 15},
        )

        logger.info(res.message)

        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.unpack_cna_mixture_params(res.x)
        )
        
        ln_state_posteriors = self.estep(res.x, initial_ln_lambdas)
        ln_lambdas = self.cna_mixture_ln_lambdas_update(ln_state_posteriors)

        ln_state_posteriors = self.estep(res.x, ln_lambdas)
        
        self.plot_rdr_baf_flat(
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=ln_state_posteriors,
            states_bag=np.c_[state_read_depths / self.realized_genome_coverage, bafs],
        )


if __name__ == "__main__":
    cna_sim = CNA_Sim()

    # cna_sim.plot_realization_flat()
    # cna_sim.plot_realization_genome()
    # cna_sim.fit_gaussian_mixture()
    
    cna_sim.fit_cna_mixture()

    print("\n\nDone.\n\n")
