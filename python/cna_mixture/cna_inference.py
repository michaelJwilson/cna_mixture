import logging

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from cna_mixture.cna_emission import CNA_emission
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.plotting import plot_rdr_baf_flat, plot_rdr_baf_genome
from cna_mixture.state_priors import CNA_categorical_prior, CNA_markov_prior
from cna_mixture.utils import param_diff

logger = logging.getLogger(__name__)


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


class CNA_inference:
    def __init__(
        self,
        num_states,
        genome_coverage,
        data,
        optimizer="L-BFGS-B",
        state_prior="categorical",
        initialize_mode="random",
        maxiter=250,
        seed=42,
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
        assert initialize_mode in ["random", "mixture_plusplus"]

        self.data = data
        self.seed = seed
        self.rng = np.random.default_rng(int(seed)) if isinstance(seed, (int, float)) else seed
        self.maxiter = maxiter
        self.optimizer = optimizer
        self.num_states = num_states
        self.num_cna_states = num_states - 1
        self.num_segments = len(data)
        self.genome_coverage = genome_coverage
        self.initialize_mode = initialize_mode

        if state_prior == "categorical":
            self.state_prior_model = CNA_categorical_prior
        elif state_prior == "markov":
            self.state_prior_model = CNA_markov_prior
        else:
            msg = f"state prior model={state_prior} is not supported."
            raise ValueError(msg)

        # NB state prior initializers log the type.
        self.state_prior_model = self.state_prior_model(
            self.num_segments,
            self.num_states,
        )

        self.emission_model = CNA_emission(
            self.num_states,
            self.genome_coverage,
            data["read_coverage"],
            data["b_reads"],
            data["snp_coverage"],
        )

        self.bounds = get_cna_mixture_bounds(self.num_states)

    @property
    def rdr(self):
        return self.data["read_coverage"] / self.genome_coverage

    @property
    def baf(self):
        return self.data["b_reads"] / self.data["snp_coverage"]

    @property
    def rdr_baf(self):
        return np.c_[self.rdr, self.baf]

    def initialize_params(self):
        """                                                                                                                                                                                                                         
        Initialize mixture parameters, state prior model given said parameters &                                                                                                                                                   
        update state priors & emissions.                                                                                                                                                                                            
        """
        # NB defines initial (BAF, RDR) for each of K states and shared overdispersions.                                                                                                                                            
        mixture_params = CNA_mixture_params(
            num_cna_states=self.num_cna_states, genome_coverage=self.genome_coverage, seed=self.seed,
        )

	# NB one "normal" state and remaining states chosen as a datapoint for copy # > 1.                                                                                                                                          
        if self.initialize_mode == "random":
            initial_cost = mixture_params.initialize_random_nonnormal_rdr_baf(self.rdr_baf)

        # NB Negative-Binomial derived read counts, b reads and snp covering reads.                                                                                                                                                 
        elif self.initialize_mode == "mixture_plusplus":
            initial_cost = mixture_params.initialize_mixture_plusplus(
                self.data["read_coverage"],
                self.data["b_reads"],
                self.data["snp_coverage"],
            )
        else:
            msg = f"{self.initialize_mode} style initialization is not supported."
            raise ValueError(msg)

        return mixture_params, initial_cost
            
    def initialize(self, **kwargs):
        """
        Initialize parameters, state prior model given said parameters &
        update state priors & emissions.
        """
        mixture_params, initial_cost = self.initialize_params()
        
        self.initial_params = mixture_params.params
        self.initial_cost = initial_cost
        
        self.last_params, self.params = None, self.initial_params
        self.last_cost, self.cost = None, self.initial_cost
        
        self.nit = 0

        if "cna_states" not in kwargs:
            kwargs["cna_states"] = mixture_params.cna_states

        if "rdr_baf" not in kwargs:
            kwargs["rdr_baf"] = self.rdr_baf

        logger.info(f"Initialized CNA states:\n{mixture_params.cna_states}\n")
            
        # BUG TODO generalizable to Markov chain?
        # NB assign ln_lambdas based on fractions hard assigned to states.
        self.state_prior_model.initialize(**kwargs)

        self.ln_state_prior = self.state_prior_model.get_ln_state_priors()
        self.ln_state_emission = self.emission_model.get_ln_state_emission_update(
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

    def pstep(self):
        """                                                                                                                                                                                                      
        Update the state prior model based on the current state posteriors,                                                                                                                                      
        and re-compute the ln_state_priors.                                                                                                                                                                      
        """
        self.state_prior_model.update(self.ln_state_posteriors)
        self.ln_state_prior = self.state_prior_model.get_ln_state_priors()

    def em_cost(self, params, verbose=False):
        """
        if state_posteriors is provided, resulting EM-cost is a lower bound to the log likelihood at
        the current params values and the assumed state_posteriors.

        NB ln_lambdas are treated independently as they are subject to a "sum to unity" constraint.
        """
        self.ln_state_emission = self.emission_model.get_ln_state_emission_update(
            params
        )

        # NB responsibilites rik, where i is the sample and k is the state.
        # NB this is *not* state-posterior weighted log-likelihood.
        # NB sum over samples and states.  Maximization -> minimization.
        cost = -(
            self.state_posteriors * (self.ln_state_prior + self.ln_state_emission)
        ).sum()

        if verbose:
            self.log_mstep(self.nit, self.last_params, self.params, params, cost)

        return cost

    def jac(self, params):
        return self.emission_model.grad_em_cost(params, self.state_posteriors)

    def post_mstep_simple(self, intermediate_result: OptimizeResult):
        """
        Callable after each M-step iteration of optimizer.  e.g. this approach                                                                                                                                                                                                  
        benefits from 'conserving' Hessian.

        Simplified form where posteriors are updated after every M.  Primarily
        for debugging / validation.
        """
        self.nit += 1

        new_params, new_cost = intermediate_result.x, intermediate_result.fun

        self.log_mstep(
            self.nit, self.last_params, self.params, new_params, new_cost
        )

        if self.nit > self.maxiter:
            logger.error(f"Failed to converge in {self.maxiter}")
            raise StopIteration

        self.ln_state_emission = self.emission_model.get_ln_state_emission_update(
            new_params
        )

        # NB update ln/state posteriors based on new emission.                                                                                                                                                                                                               
        self.estep()

        # NB update state priors based on new state posteriors.                                                                                                                                                                                                              
        self.pstep()

        # NB update ln/state posteriors based on new state priors.                                                                                                                                                                                                           
        self.estep()
    
    def post_mstep(self, intermediate_result: OptimizeResult):
        """
        Callable after each M-step iteration of optimizer.  e.g. this approach
        benefits from 'conserving' Hessian.
        """        
        # NB callback evaluated after each iteration of optimizer.
        self.nit += 1

        new_params, new_cost = intermediate_result.x, intermediate_result.fun

        self.log_mstep(
            self.nit, self.last_params, self.params, new_params, new_cost
        )

        if self.nit > self.maxiter:
            logger.error(f"Failed to converge in {self.maxiter}")
            raise StopIteration

        # NB assumed convergence tolerance.
        PARAM_FRAC_TOL = 1.0e-3
        
        # NB parameter difference across E+M step.  i.e. converged with respect to last posterior?
        #    note: relies on self.last_params == None at start of fitting, which evaluates False.
        if (pdiff := param_diff(self.last_params, new_params)) < PARAM_FRAC_TOL:
            logger.info(
                f"Converged to {100 * pdiff:.6e}% wrt last state posteriors.  Optimization complete."
            )

            logger.info(f"Parameters @ last posterior (cost={self.last_cost})=\n{self.last_params}")
            logger.info(f"Parameters @ last M (cost = {self.cost})=\n{self.params}")
            logger.info(f"Parameters @ current M (cost = {new_cost})=\n{new_params}")
            
            raise StopIteration

        # NB has parameter differences across M step converged?
        if (pdiff := param_diff(self.params, new_params)) < PARAM_FRAC_TOL:
            logger.info(
                f"Converged to {100 * pdiff:.6e}% wrt current state posteriors.  Updating posteriors."
            )

            # TODO may not be necessary?  Depends how solver calls cost (emission update) vs grad.
            self.ln_state_emission = self.emission_model.get_ln_state_emission_update(
                new_params
            )

            # NB update ln/state posteriors based on new emission.
            self.estep()

            # NB update state priors based on new state posteriors.
            self.pstep()

            # NB update ln/state posteriors based on new state priors.
            self.estep()

            self.last_params, self.last_cost = new_params.copy(), new_cost

        # NB cache current parameters to be compared at the conclusion of next M.
        self.params, self.cost = new_params.copy(), new_cost

    def fit(self):
        assert (
            self.initial_params is not None
        ), "CNA_inference.initialize(*args, **kwargs) must be called first."

        logger.info(
            f"Running {self.optimizer.upper()} optimization for {self.maxiter} max. iterations"
        )

        cost = self.em_cost(self.initial_params, verbose=True)

        self.nit = 0
        self.last_params, self.params = None, self.initial_params
        self.last_cost, self.cost = None, cost
        
        # NB see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        res = minimize(
            self.em_cost,
            self.params.copy(),
            method=self.optimizer,
            jac=self.jac,
            bounds=self.bounds,
            callback=self.post_mstep,
            constraints=None,
            options={"disp": True, "maxiter": self.maxiter},
        )

        msg = "`Successful custom convergence`" if "StopIteration" in res.message else res.message
        
        logger.info(
            f"minimization finished with message={msg} and best-fit CNA mixture params=\n{res.x}\n"
        )

        return res

    def log_mstep(self, nit, last_params, params, new_params, cost):
        state_read_depths, rdr_overdispersion, bafs, baf_overdispersion = (
            self.emission_model.unpack_params(params)
        )

        msg = f"Iteration {nit}:  Minimized cost to value: {cost:.6f} for:\n"
        msg += f"\t{self.state_prior_model}\n"
        msg += f"\tread_depths={state_read_depths}\n"
        msg += f"\tread_depth_overdispersion={rdr_overdispersion}\n"
        msg += f"\tbafs={bafs}\n"
        msg += f"\tbaf_overdispersion={baf_overdispersion}\n"
        msg += f"\tMax. frac. parameter diff. compared to last and current state posterior: {param_diff(last_params, new_params)}, {param_diff(params, new_params)}"

        logger.info(msg)
    
    def plot(self, plots_dir, params, label, title=None):
        plot_rdr_baf_flat(
            f"{plots_dir}/{label}_rdr_baf_flat.pdf",
            self.rdr,
            self.baf,
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.emission_model.get_states_bag(params),
            title=title,
        )

        plot_rdr_baf_genome(
            f"{plots_dir}/{label}_rdr_baf_genome.pdf",
            self.rdr,
            self.baf,
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.emission_model.get_states_bag(params),
            title=title,
        )
