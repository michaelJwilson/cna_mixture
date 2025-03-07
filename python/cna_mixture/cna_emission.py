import numpy as np
from cna_mixture_rs.core import (
    betabinom_logpmf,
    grad_cna_mixture_em_cost_bb_rs,
    grad_cna_mixture_em_cost_nb_rs,
    nbinom_logpmf,
)
from scipy.special import digamma
from scipy.stats import betabinom, nbinom, poisson


def reparameterize_beta_binom(bafs, overdispersion):
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
            for baf in bafs
        ]
    )

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


class CNA_emission:
    RUST_BACKEND = True

    def __init__(self, num_states, genome_coverage, ks, xs, ns):
        self.ks = ks
        self.xs = xs
        self.ns = ns

        # TODO?
        self.num_states = num_states
        self.genome_coverage = genome_coverage

    def unpack_params(self, params):
        """
        Given a cost parameter vector, unpack into named cna mixture
        parameters.
        """
        num_states = self.num_states

        # NB read_depths + overdispersion + bafs + overdispersion
        assert (
            len(params) == num_states + 1 + num_states + 1
        ), f"{params} does not satisy {num_states} states."

        state_read_depths = params[:num_states]
        rdr_overdispersion = params[num_states]

        bafs = params[num_states + 1 : 2 * num_states + 1]
        baf_overdispersion = params[2 * num_states + 1]

        return state_read_depths, rdr_overdispersion, bafs, baf_overdispersion

    def get_states_bag(self, params):
        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.unpack_params(params)
        )

        return np.c_[state_read_depths / self.genome_coverage, bafs]

    def cna_mixture_betabinom_update(self, params):
        """
        Evaluate log prob. under BetaBinom model.
        Returns (# sample, # state) array.
        """
        xs, ns = self.xs, self.ns

        _, _, bafs, baf_overdispersion = self.unpack_params(params)
        state_alpha_betas = reparameterize_beta_binom(
            bafs,
            baf_overdispersion,
        )

        if self.RUST_BACKEND:
            xs, ns = np.ascontiguousarray(xs), np.ascontiguousarray(ns)

            alphas = np.ascontiguousarray(state_alpha_betas[:, 0].copy())
            betas = np.ascontiguousarray(state_alpha_betas[:, 1].copy())

            result = betabinom_logpmf(xs, ns, betas, alphas)
            result = np.array(result)
        else:
            result = np.zeros((len(xs), len(state_alpha_betas)))

            for col, (alpha, beta) in enumerate(state_alpha_betas):
                for row, (x, n) in enumerate(zip(xs, ns)):
                    result[row, col] = betabinom.logpmf(x, n, beta, alpha)

        return result, state_alpha_betas

    def cna_mixture_nbinom_update(self, params):
        """
        Evaluate log prob. under NegativeBinom model.
        Return (# sample, # state) array.
        """
        ks = self.ks
        state_read_depths, rdr_overdispersion, _, _ = self.unpack_params(params)

        # TODO does a non-linear transform in the cost trip the optimizer?
        state_rs_ps = reparameterize_nbinom(
            state_read_depths,
            rdr_overdispersion,
        )

        if self.RUST_BACKEND:
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

    def get_ln_state_emission(self, params):
        ln_state_posterior_betabinom, _ = self.cna_mixture_betabinom_update(params)
        ln_state_posterior_nbinom, _ = self.cna_mixture_nbinom_update(params)

        return ln_state_posterior_betabinom + ln_state_posterior_nbinom

    def grad_em_cost_nb(self, params, state_posteriors):
        # TODO
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

    def grad_em_cost_bb(self, params, state_posteriors):
        # TODO
        xs = self.xs
        ns = self.ns

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

                for row, (x, n) in enumerate(zip(xs, ns)):
                    interim = zero_point + grad_ln_bb_ab_data(beta, alpha, x, n)

                    sample_grad_ps[row, col] = -tau * interim[1] + tau * interim[0]
                    sample_grad_tau[row, col] = (1.0 - baf) * interim[
                        1
                    ] + baf * interim[0]

        grad_ps = -(state_posteriors * sample_grad_ps).sum(axis=0)
        grad_tau = -(state_posteriors * sample_grad_tau).sum()

        return np.concatenate([grad_ps, np.atleast_1d(grad_tau)])

    def grad_em_cost(self, params, state_posteriors):
        return np.concatenate(
            [
                self.grad_em_cost_nb(params, state_posteriors),
                self.grad_em_cost_bb(params, state_posteriors),
            ]
        )
