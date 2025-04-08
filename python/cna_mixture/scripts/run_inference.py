import argparse
import logging
import time
import numpy as np
from pathlib import Path

from cna_mixture.cna_inference import CNA_inference
from cna_mixture.cna_sim import CNA_sim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

"""
TODOs:
  - multiple starts + best likelihood. 
  - regularizer for state overlap.
  - prior to prevent single-state occupancy.                                                                                            
"""


def run_inference(sim_dir, sim_id, state_prior, initialize_mode, seed=314, **kwargs):
    start = time.time()

    plots_dir = f"{sim_dir}/cna_sim_{sim_id}/plots/"

    Path(plots_dir).mkdir(exist_ok=True, parents=True)

    cna_sim = CNA_sim.load(sim_dir, sim_id)

    # fit_gaussian_mixture(f"{plots_dir}/gmm_rdr_baf_flat_{sim_id}.pdf", cna_sim.rdr_baf, seed=seed)

    rng = np.random.default_rng(seed)

    # NB total number of states (inc. normal).
    cna_inf = CNA_inference(
        cna_sim.num_states,
        cna_sim.genome_coverage,
        cna_sim.data,
        state_prior=state_prior,
        initialize_mode=initialize_mode,
        seed=rng,
    )

    cna_inf.initialize(**kwargs)

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
    # NB python python/cna_mixture/scripts/run_inference.py --sim-dir ~/scratch/cna_mixture/sims/ --sim-id 0
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
    parser.add_argument(
        "--state-prior",
        type=str,
        default="categorical",
        help="Assumed model for state priors.",
    )
    parser.add_argument(
        "--initialize-mode",
        type=str,
        default="random",
        help="Assumed model for initialization of state parameters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314,
        help="Seed for random number generation",
    )

    args = parser.parse_args()

    run_inference(args.sim_dir, args.sim_id, args.state_prior, args.initialize_mode, args.seed)


if __name__ == "__main__":
    main()
