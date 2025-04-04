import logging

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from cna_mixture.cna_emission import CNA_emission
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.plotting import plot_rdr_baf_flat, plot_rdr_baf_genome
from cna_mixture.state_priors import CNA_categorical_prior, CNA_markov_prior
from cna_mixture.utils import param_diff

logger = logging.getLogger(__name__)


class CNA_inference:
    def __init__(
        self,
        num_states,
        genome_coverage,
        data,
        optimizer="L-BFGS-B",
        state_prior="categorical",
        maxiter=250,
    ):
        """
        Fit CNA mixture model via Expectation Maximization.  Assumes RDR + BAF are independent
        given CNA state.

        See:
            https://udlbook.github.io/cvbook/
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        """
        # NB see e.g. https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
        assert optimizer in ["nelder-mead", "L-BFGS-B", "SLSQP"]

        self.data = data
        self.maxiter = maxiter
        self.optimizer = optimizer
        self.num_states = num_states
        self.num_segments = len(data)
        self.genome_coverage = genome_coverage

        if state_prior == "categorical":
            self.state_prior_model = CNA_categorical_prior
        elif state_prior == "markov":
            self.state_prior_model = CNA_markov_prior
        else:
            msg = f"state prior model={state_prior} is not supported."
            raise ValueError(msg)

        self.emission_model = CNA_emission(
            self.num_states,
            self.genome_coverage,
            data["read_coverage"],
            data["b_reads"],
            data["snp_coverage"],
        )

        self.bounds = self.get_cna_mixture_bounds()

    @property
    def rdr(self):
        return self.data["read_coverage"] / self.genome_coverage

    @property
    def baf(self):
        return self.data["b_reads"] / self.data["snp_coverage"]

    @property
    def rdr_baf(self):
        return np.c_[self.rdr, self.baf]

    def initialize(self, *args, **kwargs):
        # NB defines initial (BAF, RDR) for each of K states and shared overdispersions.
        mixture_params = CNA_mixture_params(
            num_cna_states=self.num_states - 1, genome_coverage=self.genome_coverage
        )

        # NB one "normal" state and remaining states chosen as a datapoint for copy # > 1.
        mixture_params.rdr_baf_choice_update(self.rdr_baf)

        logger.info(f"Initializing CNA states:\n{mixture_params.cna_states}\n")

        self.initial_params = mixture_params.params

        self.state_prior_model = self.state_prior_model(
            self.num_segments,
            self.num_states,
        )

        # NB assign ln_lambdas based on fractions hard assigned to states.
        self.state_prior_model.initialize(*args, **kwargs)

        self.ln_state_prior = self.state_prior_model.get_ln_state_priors()
        self.ln_state_emission = self.emission_model.get_ln_state_emission(
            self.initial_params
        )

        self.estep()

    def estep(self):
        """
        Calculate normalized state posteriors based on current parameter + lambda settings.
        """
        self.ln_state_posteriors = self.state_prior_model.get_ln_state_posteriors(
            self.ln_state_emission
        )
        self.state_posteriors = np.exp(self.ln_state_posteriors)

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
            self.update_message(-1, self.last_params, self.params, params, cost)

        return cost

    def jac(self, params):
        return self.emission_model.grad_em_cost(params, self.state_posteriors)

    def pstep(self):
        """
        Update the state prior model based on the current state posteriors,
        and re-compute the ln_state_priors.
        """
        self.state_prior_model.update(self.ln_state_posteriors)
        self.ln_state_prior = self.state_prior_model.get_ln_state_priors()

    def callback(self, intermediate_result: OptimizeResult):
        """
        Callable after each M-step iteration of optimizer.
        e.g. this approach benefits from 'conserving' Hessian.
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
                f"Converged to {100 * PTOL}% wrt last state posteriors.  Optimization complete."
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

            # NB update ln/state posteriors based on new emission.
            self.estep()

            # NB update state priors based on new state posteriors.
            self.pstep()

            # TODO skippable?
            # NB update ln/state posteriors based on new state priors.
            self.estep()

            self.last_params = new_params

        self.params = new_params.copy()

    def fit(self):
        assert (
            self.initial_params is not None
        ), "CNA_inference.initialize(*args, **kwargs) must be called first."

        self.last_params, self.params = None, self.initial_params
        self.nit = 0

        self.em_cost(self.params, verbose=True)

        plot_rdr_baf_flat(
            "plots/initial_rdr_baf_flat.pdf",
            self.rdr,
            self.baf,
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.emission_model.get_states_bag(self.initial_params),
            title="Initial state posteriors (based on closest state lambdas).",
        )

        logger.info(
            f"Running {self.optimizer.upper()} optimization for {self.maxiter} max. iterations"
        )

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

        plot_rdr_baf_genome(
            "plots/final_rdr_baf_genome.pdf",
            self.rdr,
            self.baf,
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.emission_model.get_states_bag(res.x),
            title="Final state posteriors",
        )

        return res

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

    def update_message(self, nit, last_params, params, new_params, cost):
        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.emission_model.unpack_params(params)
        )

        msg = f"Iteration {nit}:  Minimized cost to value: {cost:.6f} for:\n"
        msg += f"{self.state_prior_model}\nread_depths={state_read_depths}\nread_depth_overdispersion={rdr_overdispersion}\n"
        msg += f"bafs={bafs}\nbaf_overdispersion={baf_overdispersion}"
        msg += f"\nMax. frac. parameter diff. compared to last and current state posterior: {param_diff(last_params, new_params)}, {param_diff(params, new_params)}"

        logger.info(msg)
