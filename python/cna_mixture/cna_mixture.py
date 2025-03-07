import logging
import numpy as np
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.plotting import plot_rdr_baf_flat
from cna_mixture.utils import normalize_ln_posteriors, param_diff, assign_closest
from cna_mixture.state_priors import CNA_categorical_prior, CNA_markov_prior
from cna_mixture.cna_emission import CNA_emission
from scipy.optimize import minimize, check_grad, approx_fprime, OptimizeResult

logger = logging.getLogger(__name__)

class CNA_inference:
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
        self.mixture_params = CNA_mixture_params()

        self.data = data
        self.maxiter = maxiter
        self.optimizer = optimizer
        self.num_segments = len(self.baf)
        self.num_states = self.mixture_params.num_states
        self.genome_coverage = genome_coverage

        # NB one "normal" state and remaining states chosen as a datapoint for copy # > 1.
        self.mixture_params.rdr_baf_choice_update(self.rdr_baf)

        logger.info(f"Initializing CNA states:\n{self.mixture_params.cna_states}\n")

        # NB self.genome_coverage == normal_coverage currently.
        state_read_depths = self.genome_coverage * self.mixture_params.cna_states[:, 0]

        bafs = self.mixture_params.cna_states[:, 1]

        self.initial_params = np.array(
            state_read_depths.tolist()
            + [self.mixture_params.overdisp_phi]
            + bafs.tolist()
            + [self.mixture_params.overdisp_tau]
        )

        self.bounds = self.get_cna_mixture_bounds()

        self.state_prior_model = CNA_categorical_prior(
            self.mixture_params, self.rdr_baf
        )
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

        # HACK - no posterior updates.
        self.optimizer = "nelder-mead"
        self.jac = None
        self.callback = None

        res = minimize(
            self.em_cost,
            self.params.copy(),
            method=self.optimizer,
            jac=self.jac,
            bounds=self.bounds,
            callback=self.callback,
            constraints=None,
            options={"disp": True, "maxiter": self.maxiter},
        )

        logger.info(
            f"minimization finished with message={res.message} and best-fit CNA mixture params=\n{res.x}\n"
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

    def em_cost(self, params, verbose=False):
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
