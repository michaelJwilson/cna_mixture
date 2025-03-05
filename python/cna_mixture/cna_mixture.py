import logging
import numpy as np
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.plotting import plot_rdr_baf_flat
from scipy.optimize import minimize, check_grad, approx_fprime





logger = logging.getLogger(__name__)

class CNA_mixture():
    def __init__(self, rdr_baf, realized_genome_coverage, optimizer="L-BFGS-B", maxiter=100):
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

        self.rdr_baf = rdr_baf
        self.realized_genome_coverage = realized_genome_coverage

        # NB defines initial (BAF, RDR) for each of K states and shared overdispersions.                                                                                                                                                                       
        mixture_params = CNA_mixture_params()

        # NB one "normal" state and remaining states chosen as a datapoint for copy # > 1.                                                                                                                                                                     
        mixture_params.rdr_baf_choice_update(rdr_baf)

        logger.info(f"Initializing CNA states:\n{mixture_params.cna_states}\n")
        
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

        bounds = self.get_cna_mixture_bounds(rdr_baf.shape[1])

        # NB pre-populate terms to cost.
        self.ln_lambdas = self.initialize_ln_lambdas_closest(mixture_params)
        self.ln_state_prior = self.cna_mixture_categorical_update(self.ln_lambdas)
        self.ln_state_emission = self.cna_mixture_ln_emission_update(params)
        self.ln_state_posteriors = self.estep(
            self.ln_state_emission, self.ln_state_prior
        )
        self.state_posteriors = np.exp(self.ln_state_posteriors)

        cost = self.cna_mixture_em_cost(params, verbose=True)

        plot_rdr_baf_flat(
            "plots/initial_rdr_baf_flat.pdf",
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.get_states_bag(params),
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
        
        logger.info(f"Running optimization with optimizer {optimizer.upper()}")
        """
        for ii in range(maxiter):
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

        plot_rdr_baf_flat(
            "plots/final_rdr_baf_flat.pdf",
            self.rdr_baf[:, 0],
            self.rdr_baf[:, 1],
            ln_state_posteriors=self.ln_state_posteriors,
            states_bag=self.get_states_bag(params),
            title="Final state posteriors",
        )
        """
        
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

    
