import os
import logging
import argparse
import time

import numpy as np
from cna_mixture.cna_inference import CNA_inference
from cna_mixture.cna_sim import CNA_sim
from cna_mixture.fit_gaussian_mixture import fit_gaussian_mixture

np.random.seed(1234)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

"""
TODOs:

  - kmean++ like.
  - multiple starts + best likelihood. 
  - regularizer for state overlap.
  - prior to prevent single-state occupancy.                                                                                            
  - callback forward.
  - unit tests.

"""

def run_model(plots_dir):
    start = time.time()

    # NB ensure the directory exists    
    os.makedirs(plots_dir, exist_ok=True)
    
    cna_sim = CNA_sim()
    cna_sim.plot_realization_true_flat(f"{plots_dir}/truth_rdr_baf_flat.pdf")
    cna_sim.plot_realization_true_genome(f"{plots_dir}/truth_rdr_baf_genome.pdf")

    fit_gaussian_mixture(f"{plots_dir}/gmm_rdr_baf_flat.pdf", cna_sim.rdr_baf)

    # NB total number of states (inc. normal).
    cna_inf = CNA_inference(
        cna_sim.num_states,
        cna_sim.genome_coverage,
        cna_sim.data,
        state_prior="categorical",
    )
    
    cna_inf.initialize()

    cna_inf.plot(
        plots_dir,
        cna_inf.initial_params,
        "initial",
        "Initial state posteriors (based on closest state lambdas).",
    )

    res = cna_inf.fit()

    cna_inf.plot(plots_dir, res.x, "final", "Final state posteriors")

    logger.info(f"\n\nDone ({time.time() - start:.3f} seconds).\n\n")

def main():
    parser = argparse.ArgumentParser(description="Run CNA simulation and inference.")
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory to save the plots (default: 'plots').",
    )

    args = parser.parse_args()

    run_model(args.plots_dir)

if __name__ == "__main__":
    main()
    
