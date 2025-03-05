import time
import logging
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import numpy.random as random

from matplotlib.colors import LogNorm
from scipy.stats import nbinom, betabinom, poisson
from scipy.optimize import approx_fprime, check_grad, minimize
from scipy.special import logsumexp, digamma
from scipy.spatial import KDTree
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from cna_mixture_rs.core import (
    betabinom_logpmf,
    nbinom_logpmf,
    grad_cna_mixture_em_cost_nb_rs,
    grad_cna_mixture_em_cost_bb_rs,
)

np.random.seed(1234)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
RUST_BACKEND = True


def tophat_smooth(data, window_size):
    """
    Top-hat convolution of a 1D signal.
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def assign_closest(points, centers):
    """
    Assign points to the closest center.
    """
    assert len(points) > len(centers)

    tree = KDTree(centers)
    distances, idx = tree.query(points)

    return idx


def onehot_encode_states(state_array):
    """
    Given an index array of categorical states,
    return the (# samples, # states) one-hot
    encoding.

    NB equivalent to a (singular) state posterior!
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
    """
    log PDF for a Poisson distribution of given
    means and realized ks.
    """
    result = np.zeros((len(ks), len(state_mus)))

    for col, mu in enumerate(state_mus):
        for row, kk in enumerate(ks):
            result[row, col] = poisson.logpmf(kk, mu)

    return result


def normalize_ln_posteriors(ln_posteriors):
    """
    Return the normalized log posteriors.
    """
    num_samples, num_states = ln_posteriors.shape

    # NB natural logarithm by definition;
    norm = logsumexp(ln_posteriors, axis=1)
    norm = np.broadcast_to(norm.reshape(num_samples, 1), (num_samples, num_states))

    return ln_posteriors.copy() - norm


class CNA_mixture_params:
    """
    Data class for parameters required by CNA mixture model,
    with shared overdispersions.
    """

    def __init__(self):
        """
        Initialize an instance of the class with random values in the assumed bounds.
        """
        # NB normal state is treated independently
        self.num_cna_states = 3
        self.num_states = 1 + self.num_cna_states

        # NB BAF overdispersion.  Random between 25. and 55.
        self.overdisp_tau = 50.0

        # NB RDR overdispersion.  Random between 1e-2 and 4e-2
        self.overdisp_phi = 2.0e-2

        # NB list of (baf, rdr) for k=4 states, without replacement.
        integers = np.random.choice(
            np.arange(3, 10), size=self.num_cna_states, replace=False
        )
        integers = np.sort(integers)

        self.normal_state = [1.0, 0.5]
        self.cna_states = [
            [1.0 * int_sample, 1.0 / int_sample] for int_sample in integers
        ]

        self.cna_states = [self.normal_state] + self.cna_states

        self.cna_states = np.array(self.cna_states)
        self.normal_state = np.array(self.normal_state)

        self.__verify()

    def __verify(self):
        assert isinstance(
            self.cna_states, np.ndarray
        ), f"cna_states attribute must be a numpy array. Found {type(self.cna_states)}"

    def __str__(self):
        return ",  ".join([f"{key}: {value}" for key, value in self.__dict__.items()])

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

    def rdr_baf_choice_update(self, rdr_baf):
        non_normal = rdr_baf[rdr_baf[:, 0] > 1.0]

        xx = np.arange(len(non_normal))
        idx = random.choice(xx, size=self.num_states - 1, replace=False)

        self.cna_states = np.vstack([self.normal_state, non_normal[idx]])


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

        self.jump_rate_per_state = self.jump_rate / (self.num_states - 1.0)
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

        pl.savefig("plots/rdr_baf_flat.pdf")

    def plot_realization_true_flat(self):
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
        segment_index = np.arange(self.num_segments)

        figsize = (15, 10)
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)

        rdr, baf = self.rdr_baf[:, 0], self.rdr_baf[:, 1]

        smooth_rdr = tophat_smooth(rdr, window_size=100)
        smooth_baf = tophat_smooth(baf, window_size=100)

        axes[0].plot(segment_index, rdr)
        axes[0].plot(segment_index, smooth_rdr)

        axes[1].plot(segment_index, baf)
        axes[1].plot(segment_index, smooth_baf)

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

        if RUST_BACKEND:
            ks, ns = np.ascontiguousarray(ks), np.ascontiguousarray(ns)

            alphas = np.ascontiguousarray(state_alpha_betas[:, 0].copy())
            betas = np.ascontiguousarray(state_alpha_betas[:, 1].copy())

            result = betabinom_logpmf(ks, ns, betas, alphas)
            result = np.array(result)
        else:
            result = np.zeros((len(ks), len(state_alpha_betas)))

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

        if RUST_BACKEND:
            ks = np.ascontiguousarray(ks)

            rs = np.ascontiguousarray(state_rs_ps[:, 0].copy())
            ps = np.ascontiguousarray(state_rs_ps[:, 1].copy())

            result = nbinom_logpmf(ks, rs, ps)
            result = np.array(result)
        else:
            result = np.zeros((len(ks), len(state_rs_ps)))

            for col, (rr, pp) in enumerate(state_rs_ps):
                for row, kk in enumerate(ks):
                    result[row, col] = nbinom.logpmf(kk, rr, pp)

        return result, state_rs_ps

    def cna_mixture_ln_lambdas_update(self, ln_state_posteriors):
        """
        Given updated ln_state_posteriors, calculate the updated ln_lambdas.
        """
        return logsumexp(ln_state_posteriors, axis=0) - logsumexp(ln_state_posteriors)

    def cna_mixture_ln_emission_update(self, params):
        ln_state_posterior_betabinom, _ = self.cna_mixture_betabinom_update(params)
        ln_state_posterior_nbinom, _ = self.cna_mixture_nbinom_update(params)

        return ln_state_posterior_betabinom + ln_state_posterior_nbinom

    def estep(self, ln_state_emission, ln_state_prior):
        """
        Calculate normalized state posteriors based on current parameter + lambda settings.
        """
        return normalize_ln_posteriors(ln_state_emission + ln_state_prior)

    def cna_mixture_em_cost(self, params, verbose=False):
        """
        if state_posteriors is provided, resulting EM-cost is a lower bound to the log likelihood at
        the current params values and the assumed state_posteriors.

        NB ln_lambdas are treated independently as they are subject to a "sum to unity" constraint.
        """
        self.ln_state_emission = self.cna_mixture_ln_emission_update(params)

        # NB responsibilites rik, where i is the sample and k is the state.
        # NB this is *not* state-posterior weighted log-likelihood.
        # NB sum over samples and states.  Maximization -> minimization.
        cost = -(
            self.state_posteriors * (self.ln_state_prior + self.ln_state_emission)
        ).sum()

        if verbose:
            self.update_message(params, cost)

        return cost

    def grad_cna_mixture_em_cost_nb(self, params):
        state_read_depths, rdr_overdispersion, _, _ = self.unpack_cna_mixture_params(
            params
        )

        # TODO does a non-linear transform in the cost trip the optimizer?
        state_rs_ps = reparameterize_nbinom(
            state_read_depths,
            rdr_overdispersion,
        )

        ks = self.get_data_bykey("read_coverage")

        if RUST_BACKEND:
            ks = np.ascontiguousarray(ks)
            mus = np.ascontiguousarray(state_read_depths)
            rs = np.ascontiguousarray(state_rs_ps[:, 0])
            phi = rdr_overdispersion

            sample_grad_mus, sample_grad_phi = grad_cna_mixture_em_cost_nb_rs(
                ks, mus, rs, phi
            )

            sample_grad_mus = np.array(sample_grad_mus)
            sample_grad_phi = np.array(sample_grad_phi)
        else:
            sample_grad_mus = np.zeros((len(ks), len(state_rs_ps)))
            sample_grad_phi = np.zeros((len(ks), len(state_rs_ps)))

            for col, (rr, pp) in enumerate(state_rs_ps):
                mu = state_read_depths[col]
                phi = rdr_overdispersion

                zero_point = digamma(rr) / (phi * phi)
                zero_point += np.log(1.0 + phi * mu) / phi / phi
                zero_point -= phi * mu * rr / phi / (1.0 + phi * mu)

                for row, kk in enumerate(ks):
                    sample_grad_mus[row, col] = (
                        (kk - phi * mu * rr) / mu / (1.0 + phi * mu)
                    )
                    sample_grad_phi[row, col] = (
                        zero_point
                        - digamma(kk + rr) / (phi * phi)
                        + kk / phi / (1.0 + phi * mu)
                    )

        grad_mus = -(self.state_posteriors * sample_grad_mus).sum(axis=0)
        grad_phi = -(self.state_posteriors * sample_grad_phi).sum()

        return np.concatenate([grad_mus, np.atleast_1d(grad_phi)])

    def grad_cna_mixture_em_cost_bb(self, params):
        _, _, bafs, baf_overdispersion = self.unpack_cna_mixture_params(params)
        state_alpha_betas = reparameterize_beta_binom(
            bafs,
            baf_overdispersion,
        )

        ks, ns = self.get_data_bykey("b_reads"), self.get_data_bykey("snp_coverage")

        if RUST_BACKEND:
            ks = np.ascontiguousarray(ks)
            ns = np.ascontiguousarray(ns)

            alphas = np.ascontiguousarray(state_alpha_betas[:, 0])
            betas = np.ascontiguousarray(state_alpha_betas[:, 1])

            sample_grad_ps, sample_grad_tau = grad_cna_mixture_em_cost_bb_rs(
                ks, ns, alphas, betas
            )

            sample_grad_ps = np.array(sample_grad_ps)
            sample_grad_tau = np.array(sample_grad_tau)
        else:
            def grad_ln_bb_ab_zeropoint(a, b):
                gab = digamma(a + b)
                ga = digamma(a)
                gb = digamma(b)

                return np.array([gab - ga, gab - gb])

            def grad_ln_bb_ab_data(a, b, k, n):
                gka = digamma(k + a)
                gnab = digamma(n + a + b)
                gnkb = digamma(n - k + b)

                return np.array([gka - gnab, gnkb - gnab])

            ## >>>>>>>>>>>>>>>>> (ks, ns, alphas, betas)
            sample_grad_ps = np.zeros((len(ks), len(state_alpha_betas)))
            sample_grad_tau = np.zeros((len(ks), len(state_alpha_betas)))

            for col, (alpha, beta) in enumerate(state_alpha_betas):
                tau = alpha + beta
                baf = beta / tau

                zero_point = grad_ln_bb_ab_zeropoint(beta, alpha)

                for row, (k, n) in enumerate(zip(ks, ns)):
                    interim = zero_point + grad_ln_bb_ab_data(beta, alpha, k, n)

                    sample_grad_ps[row, col] = -tau * interim[1] + tau * interim[0]
                    sample_grad_tau[row, col] = (1.0 - baf) * interim[
                        1
                    ] + baf * interim[0]
            ## <<<<<<<<<<<<<<<<<

        grad_ps = -(self.state_posteriors * sample_grad_ps).sum(axis=0)
        grad_tau = -(self.state_posteriors * sample_grad_tau).sum()

        return np.concatenate([grad_ps, np.atleast_1d(grad_tau)])

    def cna_mixture_em_cost_grad(self, params, verbose=False):
        result = []
        result += self.grad_cna_mixture_em_cost_nb(params).tolist()
        result += self.grad_cna_mixture_em_cost_bb(params).tolist()

        return np.array(result)

    def update_message(self, params, cost):
        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.unpack_cna_mixture_params(params)
        )

        msg = f"Minimizing cost to value: {cost} for:\n"
        msg += f"lambdas={np.exp(self.ln_lambdas)}\nread_depths={state_read_depths}\nread_depth_overdispersion={rdr_overdispersion}\n"
        msg += f"bafs={bafs}\nbaf_overdispersion={baf_overdispersion}"

        logger.info(msg)

    def initialize_ln_lambdas_equal(self, init_mixture_params):
        return (1.0 / self.num_states) * np.ones(self.num_states)

    def initialize_ln_lambdas_closest(self, init_mixture_params):
        # TODO kmeans++ like.
        decoded_states = assign_closest(self.rdr_baf, init_mixture_params.cna_states)

        # NB categorical prior on state fractions
        _, counts = np.unique(decoded_states, return_counts=True)
        initial_ln_lambdas = np.log(counts) - np.log(np.sum(counts))

        return initial_ln_lambdas

    def get_cna_mixture_bounds(self):
        # NB exp_read_depths > 0
        bounds = [(1.0e-6, None) for _ in range(self.num_states)]

        # NB RDR overdispersion > 0
        bounds += [(1.0e-6, None)]

        # NB bafs - note, not limited to 0.5
        bounds += [(1.0e-6, 1.0) for _ in range(self.num_states)]

        # NB baf overdispersion > 0
        bounds += [(1.0e-6, None)]
        bounds = tuple(bounds)

        return bounds

    def get_cna_mixture_constraints(self):
        # TODO regularizer for state overlap?
        # NB equality constaints to be zero.
        constraints = [
            # NB sum of RDRs should explain realized genome-wide coverage.
            {
                "type": "eq",
                "fun": lambda x: np.sum(x[: self.num_states])
                - self.realized_genome_coverage,
            },
        ]

        # BUG sum of *realized* rdr values along genome should explain coverage??
        raise RuntimeError()

    def get_states_bag(self, params):
        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.unpack_cna_mixture_params(params)
        )

        return np.c_[state_read_depths / self.realized_genome_coverage, bafs]

    def fit_cna_mixture(self, optimizer="L-BFGS-B", maxiter=100):
        """
        Fit CNA mixture model via Expectation Maximization.
        Assumes RDR + BAF are independent given CNA state.

        See:
            https://udlbook.github.io/cvbook/

            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        """
        # NB see e.g. https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
        assert optimizer in ["nelder-mead", "L-BFGS-B", "SLSQP"]

        # NB defines initial (BAF, RDR) for each of K states and shared overdispersions.
        mixture_params = CNA_mixture_params()
        mixture_params.rdr_baf_choice_update(self.rdr_baf)

        logging.info(f"Initializing CNA states:\n{mixture_params.cna_states}\n")

        # NB self.realized_genome_coverage == normal_coverage currently.
        state_read_depths = (
            self.realized_genome_coverage * mixture_params.cna_states[:, 0]
        )

        bafs = mixture_params.cna_states[:, 1]

        params = np.array(
            state_read_depths.tolist()
            + [mixture_params.overdisp_phi]
            + bafs.tolist()
            + [mixture_params.overdisp_tau]
        )

        bounds = self.get_cna_mixture_bounds()

        self.ln_lambdas = self.initialize_ln_lambdas_closest(mixture_params)
        self.ln_state_emission = self.cna_mixture_ln_emission_update(params)
        self.ln_state_prior = self.cna_mixture_categorical_update(self.ln_lambdas)
        self.ln_state_posteriors = self.estep(
            self.ln_state_emission, self.ln_state_prior
        )
        self.state_posteriors = np.exp(self.ln_state_posteriors)

        cost = self.cna_mixture_em_cost(params, verbose=True)

        # TODO test
        em_cost_grad = self.cna_mixture_em_cost_grad(params)
        approx_grad = approx_fprime(
            params, self.cna_mixture_em_cost, np.sqrt(np.finfo(float).eps)
        )

        err = check_grad(
            self.cna_mixture_em_cost, self.cna_mixture_em_cost_grad, params
        )

        print(em_cost_grad)
        print(approx_grad)
        
        assert err < 1.0, f"{err}"

        for ii in range(maxiter):
            # TODO prior to prevent single-state occupancy.
            # TODO callback forward.
            res = minimize(
                self.cna_mixture_em_cost,
                params,
                method=optimizer,
                jac=self.cna_mixture_em_cost_grad,
                bounds=bounds,
                constraints=None,
                options={"disp": True, "maxiter": 5},
            )

            max_frac_shift = np.max(np.abs((1.0 - res.x / params)))

            params, cost = res.x, res.fun

            self.ln_state_posteriors = self.estep(
                self.ln_state_emission, self.ln_state_prior
            )

            self.state_posteriors = np.exp(self.ln_state_posteriors)

            self.ln_lambdas = self.cna_mixture_ln_lambdas_update(
                self.ln_state_posteriors
            )

            self.ln_state_prior = self.cna_mixture_categorical_update(self.ln_lambdas)

            logger.info(
                f"minimization success={res.success}, with max parameter frac. update: {max_frac_shift} and message={res.message}\n"
            )

            self.update_message(params, cost)

            if max_frac_shift < 0.01:
                break

        logger.info(f"Found best-fit CNA mixture params:\n{params}")

        # title="Initial state posteriors (based on closest state lambdas).",
        self.plot_rdr_baf_flat(
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.get_states_bag(params),
        )


def main():
    start = time.time()
    cna_sim = CNA_Sim()

    # cna_sim.plot_realization_true_flat()
    # cna_sim.plot_realization_genome()
    # cna_sim.fit_gaussian_mixture()

    cna_sim.fit_cna_mixture()

    print(f"\n\nDone ({time.time() - start:.3f} seconds).\n\n")


if __name__ == "__main__":
    main()
