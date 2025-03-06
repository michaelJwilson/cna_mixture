import logging
import numpy as np
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.plotting import plot_rdr_baf_flat
from cna_mixture.utils import normalize_ln_posteriors
from cna_mixture.negative_binomial import reparameterize_nbinom
from cna_mixture.beta_binomial import reparameterize_beta_binom
from scipy.spatial import KDTree
from scipy.optimize import minimize, check_grad, approx_fprime, OptimizeResult
from scipy.special import digamma, logsumexp
from cna_mixture_rs.core import (
    betabinom_logpmf,
    nbinom_logpmf,
    grad_cna_mixture_em_cost_nb_rs,
    grad_cna_mixture_em_cost_bb_rs,
)


logger = logging.getLogger(__name__)

def assign_closest(points, centers):
    """
    Assign points to the closest center.
    """
    assert len(points) > len(centers)

    tree = KDTree(centers)
    distances, idx = tree.query(points)

    return idx

class CNA_mixture:
    # NB call the rust backend.
    RUST_BACKEND = True
    
    def __init__(
        self, data, rdr_baf, realized_genome_coverage, optimizer="L-BFGS-B", maxiter=100
    ):
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

	# NB one "normal" state and remaining states chosen as a datapoint for copy # > 1.
        mixture_params.rdr_baf_choice_update(rdr_baf)

        logger.info(f"Initializing CNA states:\n{mixture_params.cna_states}\n")

        self.data = data
        self.rdr_baf = rdr_baf
        self.maxiter = maxiter
        self.optimizer = optimizer
        self.num_segments = len(rdr_baf)
        self.num_states = mixture_params.num_states
        self.realized_genome_coverage = realized_genome_coverage

        # NB self.realized_genome_coverage == normal_coverage currently.
        state_read_depths = (
            self.realized_genome_coverage * mixture_params.cna_states[:, 0]
        )

        bafs = mixture_params.cna_states[:, 1]

        self.initial_params = np.array(
            state_read_depths.tolist()
            + [mixture_params.overdisp_phi]
            + bafs.tolist()
            + [mixture_params.overdisp_tau]
        )

        self.bounds = self.get_cna_mixture_bounds(self.num_states)
        
        # NB pre-populate terms to cost.
        self.ln_lambdas = self.initialize_ln_lambdas_closest(mixture_params)
        self.ln_state_prior = self.cna_mixture_categorical_update(self.ln_lambdas)
        self.ln_state_emission = self.cna_mixture_ln_emission_update(self.initial_params)

        self.estep(
            self.ln_state_emission, self.ln_state_prior
        )
        
        self.state_posteriors = np.exp(self.ln_state_posteriors)

        cost = self.cna_mixture_em_cost(self.initial_params)

        plot_rdr_baf_flat(
            "plots/initial_rdr_baf_flat.pdf",
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.get_states_bag(self.initial_params),
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

    @staticmethod
    def param_diff(params, new_params):
        if params is None:
            return np.inf
        elif new_params is None:
            return np.inf
        else:
            return np.max(np.abs((1.0 - new_params / params)))
        
    def callback(self, intermediate_result: OptimizeResult):
        """
        Callable after each iteration of optimizer.  e.g. benefits from preserving Hessian.
        """
        PTOL = 1.e-2
        
        self.nit += 1

        if self.nit > self.maxiter:
            logger.info(f"Failed to converge in {self.maxiter}")
            raise StopIteration
        
        new_params, new_cost = intermediate_result.x, intermediate_result.fun

        self.update_message(self.nit, self.last_params, self.params, new_params, new_cost)
       
        # NB converged with respect to last posterior?
        if self.param_diff(self.last_params, new_params) < PTOL:
            logger.info(f"Converged to {100 * PTOL}% wrt last state posteriors.  Complete.")                
            raise StopIteration

        if (self.param_diff(self.params, new_params) < PTOL):
            logger.info(f"Converged to {100 * PTOL}% wrt current state posteriors.  Updating posteriors.")

            # TODO may not be necessary?  Depends how solver calls cost (emission update) vs grad.                                                                                                                   
            self.ln_state_emission = self.cna_mixture_ln_emission_update(new_params)

            self.estep(self.ln_state_emission, self.ln_state_prior)

            self.last_params = new_params

        self.params = new_params.copy()
            
    def fit(self):
        logger.info(f"Running optimization with optimizer {self.optimizer.upper()}")

        self.last_params, self.params = None, self.initial_params
        self.nit = 0

        res = minimize(
            self.cna_mixture_em_cost,
            self.params.copy(),
            method=self.optimizer,
            jac=self.cna_mixture_em_cost_grad,
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
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.get_states_bag(res.x),
            title="Final state posteriors",
        )

    @staticmethod
    def get_cna_mixture_bounds(num_states):
        # NB exp_read_depths > 0
        bounds = [(1.0e-6, None) for _ in range(num_states)]

        # NB RDR overdispersion > 0
        bounds += [(1.0e-6, None)]

        # NB bafs - note, not limited to 0.5
        bounds += [(1.0e-6, 1.0) for _ in range(num_states)]

        # NB baf overdispersion > 0
        bounds += [(1.0e-6, None)]

        return tuple(bounds)

    def get_cna_mixture_constraints(self):
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

    def unpack_cna_mixture_params(self, params):
        """
        Given a cost parameter vector, unpack into named cna mixture
        parameters.
        """
        num_states = self.num_states

        # NB read_depths + overdispersion + bafs + overdispersion
        assert len(params) == num_states + 1 + num_states + 1, f"{params} does not satisy {num_states} states."

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

        ks, ns = self.data["b_reads"], self.data["snp_coverage"]

        if self.RUST_BACKEND:
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

        ks = self.data["read_coverage"]

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
        self.ln_state_posteriors = normalize_ln_posteriors(ln_state_emission + ln_state_prior)

        self.state_posteriors = np.exp(
            self.ln_state_posteriors
        )

        self.ln_lambdas = self.cna_mixture_ln_lambdas_update(
            self.ln_state_posteriors
        )

        self.ln_state_prior = self.cna_mixture_categorical_update(self.ln_lambdas)

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
            self.update_message(-1, params, cost)

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

        ks = self.data["read_coverage"]

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

        grad_mus = -(self.state_posteriors * sample_grad_mus).sum(axis=0)
        grad_phi = -(self.state_posteriors * sample_grad_phi).sum()

        return np.concatenate([grad_mus, np.atleast_1d(grad_phi)])

    def grad_cna_mixture_em_cost_bb(self, params):
        _, _, bafs, baf_overdispersion = self.unpack_cna_mixture_params(params)
        state_alpha_betas = reparameterize_beta_binom(
            bafs,
            baf_overdispersion,
        )

        ks, ns = self.data["b_reads"], self.data["snp_coverage"]

        if self.RUST_BACKEND:
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

        grad_ps = -(self.state_posteriors * sample_grad_ps).sum(axis=0)
        grad_tau = -(self.state_posteriors * sample_grad_tau).sum()

        return np.concatenate([grad_ps, np.atleast_1d(grad_tau)])

    def cna_mixture_em_cost_grad(self, params, verbose=False):
        result = []
        result += self.grad_cna_mixture_em_cost_nb(params).tolist()
        result += self.grad_cna_mixture_em_cost_bb(params).tolist()

        return np.array(result)

    def update_message(self, nit, last_params, params, new_params, cost):
        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.unpack_cna_mixture_params(params)
        )

        msg = f"Iteration {nit}:  Minimizing cost to value: {cost} for:\n"
        msg += f"lambdas={np.exp(self.ln_lambdas)}\nread_depths={state_read_depths}\nread_depth_overdispersion={rdr_overdispersion}\n"
        msg += f"bafs={bafs}\nbaf_overdispersion={baf_overdispersion}"
        msg += f"\nMax. frac. parameter diff. compared to last and current state posterior: {self.param_diff(last_params, new_params)}, {self.param_diff(params, new_params)}"
        
        logger.info(msg)

    def initialize_ln_lambdas_equal(self, init_mixture_params):
        return (1.0 / self.num_states) * np.ones(self.num_states)

    def initialize_ln_lambdas_closest(self, init_mixture_params):
        decoded_states = assign_closest(self.rdr_baf, init_mixture_params.cna_states)

        # NB categorical prior on state fractions
        _, counts = np.unique(decoded_states, return_counts=True)
        initial_ln_lambdas = np.log(counts) - np.log(np.sum(counts))

        return initial_ln_lambdas

    def get_states_bag(self, params):
        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.unpack_cna_mixture_params(params)
        )

        return np.c_[state_read_depths / self.realized_genome_coverage, bafs]
