import logging
import numpy as np
from cna_mixture_rs.core import (
    nbinom_rs,
    betabinom_rs,
    CnaEmissionRs,
    CnaEmissionCompressedRs,
)
from scipy.special import digamma
from scipy.stats import betabinom, nbinom, poisson

logger = logging.getLogger(__name__)


def reparameterize_beta_binom(bafs, overdispersion):
    """
    Given the array of BAFs for all states and a shared overdispersion,
    return the (# states, 2) array of [alpha, beta] for each state,
    where beta is associated to the BAF probability.
    """
    interim = np.array(
        [
            [
                (1.0 - baf) * overdispersion,
                baf * overdispersion,
            ]
            for baf in bafs
        ]
    )

    # NB alphas, betas
    return np.ravel(interim[:, 0]), np.ravel(interim[:, 1])


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

    return np.ravel(rs), np.ravel(ps)


class CNA_emission_backed_rs:
    # NB patch class that handles bafs -> alphas, betas + delegates.
    def __init__(self, num_states, ks, xs, bs, ns, ws=None):
        # NB ks are NB derived.  xs (exposure) == T_n x lambda_g.
        self.ks = ks.copy()
        self.xs = xs.copy()

        # NB bs and ns are BB derived.
        self.bs = bs.copy()
        self.ns = ns.copy()

        self.num_states = num_states

        self.engine = CnaEmissionRs(
            num_states,
            self.ks,
            self.xs,
            self.bs,
            self.ns,
        )
        
        self.engine_compressed = CnaEmissionCompressedRs(
            num_states,
            self.ks,
            self.xs,
            self.bs,
            self.ns,
        )

        if ws is not None:
            self.engine_compressed.update_weights(ws)

        logger.info("Initialized rust emission class.")

    def update_weights(self, weights):
        self.engine_compressed.update_weights(weights)
        
    def nbinom(self, rdrs, rdr_overdispersion):        
        return self.engine.nbinom(rdrs, rdr_overdispersion)

    def nbinom_reduce(self, rdrs, rdr_overdispersion):
        return self.engine_compressed.nbinom(rdrs, rdr_overdispersion)

    def betabinom(self, bafs, baf_overdispersion):        
        alphas, betas = reparameterize_beta_binom(bafs, baf_overdispersion)
        return self.engine.betabinom(betas, alphas)

    def betabinom_reduce(self, bafs, baf_overdispersion):
        alphas, betas = reparameterize_beta_binom(bafs, baf_overdispersion)
        return self.engine_compressed.betabinom_reduce(betas, alphas)

    def emission(self, rdrs, rdr_overdispersion, bafs, baf_overdispersion):        
        alphas, betas = reparameterize_beta_binom(bafs, baf_overdispersion)
        
        # NB assumes independent
        return self.engine.nbinom(rdrs, rdr_overdispersion) + self.engine.betabinom(betas, alphas)

    def emission_reduce(self, rdrs, rdr_overdispersion, bafs, baf_overdispersion):
        alphas, betas = reparameterize_beta_binom(bafs, baf_overdispersion)
        
        # NB assumes independent
        return self.engine_compressed.nbinom_reduce(rdrs, rdr_overdispersion) + self.engine_compressed.betabinom_reduce(betas, alphas)


class CNA_emission_backend:
    """
    python equivalent validation class for CnaEmissionRs.
    """

    def __init__(self, num_states, ks, xs, bs, ns, ws=None):
        # NB ks are NB derived.  xs (exposure) == T_n x lambda_g.
        self.ks = ks
        self.xs = xs

        # NB bs and ns are BB derived.
        self.bs = bs
        self.ns = ns

        self.ws = np.ones((len(ks), num_states), dtype=float) if ws is None else ws

        self.num_states = num_states

        logger.info("Initialized (python) validation emission class.")

    def update_weights(self, ws):
        self.ws = ws
        
    def nbinom(self, rdrs, rdr_overdispersion):
        """
        Evaluate log prob. under NegativeBinom model.
        Return (# sample, # state) array.
        """
        result = np.zeros((len(self.ks), len(rdrs)))

        for col, mm in enumerate(rdrs):
            for row, (kk, xx) in enumerate(zip(self.ks, self.xs)):
                rr, pp = reparameterize_nbinom(
                    xx * mm,
                    rdr_overdispersion,
                )

                result[row, col] = nbinom.logpmf(kk, rr, pp)

        return result

    def nbinom_reduce(self, rdrs, rdr_overdispersion):
        result = self.cna_mixture_nbinom_update(rdrs, rdr_overdispersion)

        return (self.ws * result).sum()

    def betabinom(self, bafs, baf_overdispersion):
        """
        Evaluate log prob. under BetaBinom model given model parameter vector.
        Returns (# sample, # state) array.
        """
        alphas, betas = reparameterize_beta_binom(bafs, baf_overdispersion)
        result = np.zeros((len(self.bs), len(alphas)))

        for col, (alpha, beta) in enumerate(zip(alphas, betas)):
            for row, (b, n) in enumerate(zip(self.bs, self.ns, strict=False)):
                result[row, col] = betabinom.logpmf(b, n, beta, alpha)

        return result

    def betabinom_reduce(self, bafs, baf_overdispersion):
        result = self.cna_mixture_betabinom_update(bafs, baf_overdispersion)
        return (self.ws * result).sum()

    def emission(self, rdrs, rdr_overdispersion, bafs, baf_overdispersion):
        # NB assumes independent
        return self.nbinom(rdrs, rdr_overdispersion) + self.betabinom(
            bafs, baf_overdispersion
        )

    def emission_reduce(self, rdrs, rdr_overdispersion, bafs, baf_overdispersion):
        # NB assumes independent
        return self.nbinom_reduce(rdrs, rdr_overdispersion) + self.betabinom_reduce(
            bafs, baf_overdispersion
        )


class CNA_emission:
    def __init__(
        self, num_states, ks, xs, bs, ns, ws=None, backend="rust", compress=True
    ):
        self.length = len(ks)
        self.num_states = num_states
        
        if backend == "rust":
            self.backend = CNA_emission_backed_rs(num_states, ks, xs, bs, ns, ws)
        else:            
            self.backend = CNA_emission_backend(num_states, ks, xs, bs, ns, ws)

    @property
    def ks(self):
        return self.backend.ks

    @property
    def	xs(self):
        return self.backend.xs

    @property
    def	bs(self):
        return self.backend.bs

    @property
    def	ns(self):
        return self.backend.ns
    
    def __len__(self):
        return len(self.ks)
            
    def unpack_params(self, params):
        """
        Given a cost parameter vector, unpack into named cna mixture
        parameters.
        """
        # NB read_depths + overdispersion + bafs + overdispersion
        assert (
            len(params) == self.num_states + 1 + self.num_states + 1
        ), f"{params} does not satisy {self.num_states} states."

        num_states = self.num_states

        rdrs = params[:num_states]
        rdr_overdispersion = params[num_states]

        bafs = params[num_states + 1 : 2 * num_states + 1]
        baf_overdispersion = params[2 * num_states + 1]

        return rdrs, rdr_overdispersion, bafs, baf_overdispersion

    def get_states_bag(self, params):
        rdrs, rdr_overdispersion, bafs, baf_overdispersion = self.unpack_params(params)
        return np.c_[rdrs, bafs]

    def update_weights(self):
        self.backend.update_weights(ws)
    
    def nbinom(self, params):
        rdrs, rdr_overdispersion, *_ = self.unpack_params(params)
        return self.backend.nbinom(rdrs, rdr_overdispersion)

    def nbinom_reduce(self, params):
        rdrs, rdr_overdispersion, *_ = self.unpack_params(params)
        return self.backend.nbinom_reduce(rdrs, rdr_overdispersion)

    def betabinom(self, params):
        *_, bafs, baf_overdispersion = self.unpack_params(params)
        return self.backend.betabinom(bafs, baf_overdispersion)

    def betabinom_reduce(self, params):
        *_, bafs, baf_overdispersion = self.unpack_params(params)
        return self.backend.betabinom_reduce(bafs, baf_overdispersion)

    def emission(self, params):
        rdrs, rdr_overdispersion, bafs, baf_overdispersion = self.unpack_params(params)
        return self.backend.emission(rdrs, rdr_overdispersion, bafs, baf_overdispersion)

    def emission_reduce(self, params):
        rdrs, rdr_overdispersion, bafs, baf_overdispersion = self.unpack_params(params)
        return self.backend.emission_reduce(rdrs, rdr_overdispersion, bafs, baf_overdispersion)

    """
    def grad_em_cost_nb(self, params, state_posteriors):
        ks = self.ks
        state_read_depths, rdr_overdispersion, _, _ = self.unpack_params(params)

        # TODO does a non-linear transform in the cost trip the optimizer?
        state_rs_ps = reparameterize_nbinom(
            state_read_depths,
            rdr_overdispersion,
        )

        if self.RUST_BACKEND:
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

            for col, (rr, _) in enumerate(state_rs_ps):
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

        grad_mus = -(state_posteriors * sample_grad_mus).sum(axis=0)
        grad_phi = -(state_posteriors * sample_grad_phi).sum()

        return np.concatenate([grad_mus, np.atleast_1d(grad_phi)])
    """
    """
    def grad_em_cost_bb(self, params, state_posteriors):
        xs, ns = self.xs, self.ns

        _, _, bafs, baf_overdispersion = self.unpack_params(params)
        state_alpha_betas = reparameterize_beta_binom(
            bafs,
            baf_overdispersion,
        )

        if self.RUST_BACKEND:
            xs = np.ascontiguousarray(xs)
            ns = np.ascontiguousarray(ns)

            alphas = np.ascontiguousarray(state_alpha_betas[:, 0])
            betas = np.ascontiguousarray(state_alpha_betas[:, 1])

            sample_grad_ps, sample_grad_tau = grad_cna_mixture_em_cost_bb_rs(
                xs, ns, alphas, betas
            )

            sample_grad_ps = np.array(sample_grad_ps)
            sample_grad_tau = np.array(sample_grad_tau)
        else:

            def grad_ln_bb_ab_zeropoint(a, b):
                gab = digamma(a + b)
                ga = digamma(a)
                gb = digamma(b)

                return np.array([gab - ga, gab - gb])

            def grad_ln_bb_ab_data(a, b, x, n):
                gxa = digamma(x + a)
                gnab = digamma(n + a + b)
                gnxb = digamma(n - x + b)

                return np.array([gxa - gnab, gnxb - gnab])

            sample_grad_ps = np.zeros((len(xs), len(state_alpha_betas)))
            sample_grad_tau = np.zeros((len(xs), len(state_alpha_betas)))

            for col, (alpha, beta) in enumerate(state_alpha_betas):
                tau = alpha + beta
                baf = beta / tau

                zero_point = grad_ln_bb_ab_zeropoint(beta, alpha)

                for row, (x, n) in enumerate(zip(xs, ns, strict=False)):
                    interim = zero_point + grad_ln_bb_ab_data(beta, alpha, x, n)

                    sample_grad_ps[row, col] = -tau * interim[1] + tau * interim[0]
                    sample_grad_tau[row, col] = (1.0 - baf) * interim[
                        1
                    ] + baf * interim[0]

        grad_ps = -(state_posteriors * sample_grad_ps).sum(axis=0)
        grad_tau = -(state_posteriors * sample_grad_tau).sum()

        return np.concatenate([grad_ps, np.atleast_1d(grad_tau)])
    """
    """
    def grad_em_cost(self, params, state_posteriors, production_mode=True):
        if not production_mode:
            # HACK *slow* guard against log probs.
            assert np.all(state_posteriors >= 0.0)

        return np.concatenate(
            [
                self.grad_em_cost_nb(params, state_posteriors),
                self.grad_em_cost_bb(params, state_posteriors),
            ]
        )
    """
