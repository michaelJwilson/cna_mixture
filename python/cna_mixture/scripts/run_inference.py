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


def run_inference(sim_dir, sim_id=0):
    start = time.time()

    plots_dir = f"{sim_dir}/cna_sim_{sim_id}/plots/"

    os.makedirs(plots_dir, exist_ok=True)

    cna_sim = CNA_sim.load(sim_dir, sim_id)
    
    fit_gaussian_mixture(f"{plots_dir}/gmm_rdr_baf_flat_{sim_id}.pdf", cna_sim.rdr_baf)

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

    logger.info(f"Done ({time.time() - start:.3f} seconds).\n\n")


def main():
    # NB python python/cna_mixture/scripts/run_inference.py --sim-dir ~/scratch/cna_mixture/sims/ --sim_id=0
    parser = argparse.ArgumentParser(description="Run CNA inference.")
    parser.add_argument(
        "--sim-dir",
        type=str,
        default="plots",
        help="Directory to simulation outputs",
    )
    parser.add_argument(
        "--sim-id",
        type=int,
        default=0,
        help="Simulation ID",
    )

    args = parser.parse_args()

    run_inference(args.sim_dir, args.sim_id)


if __name__ == "__main__":
    main()
