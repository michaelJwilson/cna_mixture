import logging
import numpy as np
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.plotting import plot_rdr_baf_flat
from cna_mixture.utils import normalize_ln_posteriors, param_diff, assign_closest
from cna_mixture.negative_binomial import reparameterize_nbinom
from cna_mixture.beta_binomial import reparameterize_beta_binom
from scipy.optimize import minimize, check_grad, approx_fprime, OptimizeResult
from scipy.special import digamma, logsumexp
from cna_mixture_rs.core import (
    betabinom_logpmf,
    nbinom_logpmf,
    grad_cna_mixture_em_cost_nb_rs,
    grad_cna_mixture_em_cost_bb_rs,
)


logger = logging.getLogger(__name__)


class CNA_categorical_prior:
    def __init__(self, mixture_params, rdr_baf):
        self.num_states = mixture_params.num_states
        self.cna_states = mixture_params.cna_states
        self.ln_lambdas = self.ln_lambdas_closest(rdr_baf, self.cna_states)

    @staticmethod
    def ln_lambdas_equal(num_states):
        return (1.0 / num_states) * np.ones(num_states)

    @staticmethod
    def ln_lambdas_closest(rdr_baf, cna_states):
        decoded_states = assign_closest(rdr_baf, cna_states)

        # NB categorical prior on state fractions
        _, counts = np.unique(decoded_states, return_counts=True)
        ln_lambdas = np.log(counts) - np.log(np.sum(counts))

        return ln_lambdas

    def update(self, ln_state_posteriors):
        """
        Given updated ln_state_posteriors, calculate the updated ln_lambdas.
        """
        self.ln_lambdas = logsumexp(ln_state_posteriors, axis=0) - logsumexp(
            ln_state_posteriors
        )

    def get_state_priors(self, num_segments):
        """
        Broadcast per-state categorical priors to equivalent (samples x state)
        Prior array.
        """
        ln_norm = logsumexp(self.ln_lambdas)

        # NB ensure normalized.
        return np.broadcast_to(
            self.ln_lambdas - ln_norm, (num_segments, len(self.ln_lambdas))
        ).copy()

    def __str__(self):
        return f"lambdas={np.exp(self.ln_lambdas)}"

class CNA_markov_prior:
    def __init__(self):
        raise NotImplementedError()


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


class CNA_mixture:
    def __init__(self, genome_coverage, data, optimizer="L-BFGS-B", maxiter=100):
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

        self.data = data
        self.maxiter = maxiter
        self.optimizer = optimizer
        self.num_segments = len(self.baf)
        self.num_states = mixture_params.num_states
        self.genome_coverage = genome_coverage

        # NB one "normal" state and remaining states chosen as a datapoint for copy # > 1.
        mixture_params.rdr_baf_choice_update(self.rdr_baf)

        logger.info(f"Initializing CNA states:\n{mixture_params.cna_states}\n")

        # NB self.genome_coverage == normal_coverage currently.
        state_read_depths = self.genome_coverage * mixture_params.cna_states[:, 0]

        bafs = mixture_params.cna_states[:, 1]

        self.initial_params = np.array(
            state_read_depths.tolist()
            + [mixture_params.overdisp_phi]
            + bafs.tolist()
            + [mixture_params.overdisp_tau]
        )

        self.bounds = self.get_cna_mixture_bounds()

        self.state_prior_model = CNA_categorical_prior(mixture_params, self.rdr_baf)
        self.emission_model = CNA_emission(
            self.num_states,
            self.genome_coverage,
            data["read_coverage"],
            data["b_reads"],
            data["snp_coverage"],
        )

        # NB pre-populate terms to cost.
        self.ln_state_prior = self.state_prior_model.get_state_priors(self.num_segments)
        self.ln_state_emission = self.emission_model.get_ln_state_emission(
            self.initial_params
        )

        self.estep(self.ln_state_emission, self.ln_state_prior)

        plot_rdr_baf_flat(
            "plots/initial_rdr_baf_flat.pdf",
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.emission_model.get_states_bag(self.initial_params),
            title="Initial state posteriors (based on closest state lambdas).",
        )

        """
        # TODO test
        em_cost_grad = self.cna_mixture_em_cost_grad(params)
        approx_grad = approx_fprime(
            params, self.cna_mixture_em_cost, np.sqrt(np.finfo(float).eps)
        )

        err = check_grad(
            self.cna_mixture_em_cost, self.cna_mixture_em_cost_grad, params
        )

        # NB expect ~0.77
        assert err < 1.0, f"{err}"
        """

    @property
    def rdr(self):
        return self.data["read_coverage"] / self.genome_coverage

    @property
    def baf(self):
        return self.data["b_reads"] / self.data["snp_coverage"]

    @property
    def rdr_baf(self):
        return np.c_[self.rdr, self.baf]

    def callback(self, intermediate_result: OptimizeResult):
        """
        Callable after each iteration of optimizer.  e.g. benefits from preserving Hessian.
        """
        # NB assumed convergence tolerance for *fractional* change in parameter.
        PTOL = 1.0e-3

        self.nit += 1

        new_params, new_cost = intermediate_result.x, intermediate_result.fun

        self.update_message(
            self.nit, self.last_params, self.params, new_params, new_cost
        )

        if self.nit > self.maxiter:
            logger.info(f"Failed to converge in {self.maxiter}")
            raise StopIteration

        # NB converged with respect to last posterior?
        if param_diff(self.last_params, new_params) < PTOL:
            logger.info(
                f"Converged to {100 * PTOL}% wrt last state posteriors.  Complete."
            )
            raise StopIteration

        if param_diff(self.params, new_params) < PTOL:
            logger.info(
                f"Converged to {100 * PTOL}% wrt current state posteriors.  Updating posteriors."
            )

            # TODO may not be necessary?  Depends how solver calls cost (emission update) vs grad.
            self.ln_state_emission = self.emission_model.get_ln_state_emission(
                new_params
            )

            self.estep(self.ln_state_emission, self.ln_state_prior)

            self.pstep()
            
            self.estep(self.ln_state_emission, self.ln_state_prior)

            self.last_params = new_params

        self.params = new_params.copy()

    def jac(self, params):
        return self.emission_model.grad_em_cost(params, self.state_posteriors)
        
    def fit(self):
        logger.info(f"Running optimization with optimizer {self.optimizer.upper()}")

        self.last_params, self.params = None, self.initial_params
        self.nit = 0

        res = minimize(
            self.cna_mixture_em_cost,
            self.params.copy(),
            method=self.optimizer,
            jac=self.jac,
            bounds=self.bounds,
            callback=self.callback,
            constraints=None,
            options={"disp": True, "maxiter": self.maxiter},
        )

        logger.info(
            f"minimization success with best-fit CNA mixture params=\n{res.x}\n"
        )

        plot_rdr_baf_flat(
            "plots/final_rdr_baf_flat.pdf",
            self.rdr,
            self.baf,
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.emission_model.get_states_bag(res.x),
            title="Final state posteriors",
        )

    def get_cna_mixture_bounds(self):
        # NB exp_read_depths > 0
        bounds = [(1.0e-6, None) for _ in range(self.num_states)]

        # NB RDR overdispersion > 0
        bounds += [(1.0e-6, None)]

        # NB bafs - note, not limited to 0.5
        bounds += [(1.0e-6, 1.0) for _ in range(self.num_states)]

        # NB baf overdispersion > 0
        bounds += [(1.0e-6, None)]

        return tuple(bounds)

    def estep(self, ln_state_emission, ln_state_prior):
        """
        Calculate normalized state posteriors based on current parameter + lambda settings.
        """
        self.ln_state_posteriors = normalize_ln_posteriors(
            ln_state_emission + ln_state_prior
        )

        self.state_posteriors = np.exp(self.ln_state_posteriors)

    def pstep(self):
        self.state_prior_model.update(self.ln_state_posteriors)
        self.ln_state_prior = self.state_prior_model.get_state_priors(self.num_segments)

    def cna_mixture_em_cost(self, params, verbose=False):
        """
        if state_posteriors is provided, resulting EM-cost is a lower bound to the log likelihood at
        the current params values and the assumed state_posteriors.

        NB ln_lambdas are treated independently as they are subject to a "sum to unity" constraint.
        """
        self.ln_state_emission = self.emission_model.get_ln_state_emission(params)

        # NB responsibilites rik, where i is the sample and k is the state.
        # NB this is *not* state-posterior weighted log-likelihood.
        # NB sum over samples and states.  Maximization -> minimization.
        cost = -(
            self.state_posteriors * (self.ln_state_prior + self.ln_state_emission)
        ).sum()

        if verbose:
            # TODO
            self.update_message(-1, params, cost)

        return cost

    def update_message(self, nit, last_params, params, new_params, cost):
        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.emission_model.unpack_params(params)
        )

        msg = f"Iteration {nit}:  Minimizing cost to value: {cost} for:\n"
        msg += f"{self.state_prior_model}\nread_depths={state_read_depths}\nread_depth_overdispersion={rdr_overdispersion}\n"
        msg += f"bafs={bafs}\nbaf_overdispersion={baf_overdispersion}"
        msg += f"\nMax. frac. parameter diff. compared to last and current state posterior: {param_diff(last_params, new_params)}, {param_diff(params, new_params)}"

        logger.info(msg)
