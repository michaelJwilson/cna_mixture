import os
import logging
import argparse
import time

import numpy as np
from cna_mixture.cna_inference import CNA_inference
from cna_mixture.cna_sim import CNA_sim
from cna_mixture.fit_gaussian_mixture import fit_gaussian_mixture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def run_sim(output_dir, num_sims=1, seed=314):
    start = time.time()
    
    for num_sim in range(num_sims):
        # NB ensure the directory exists
        os.makedirs(f"{output_dir}/cna_sim_{num_sim}/plots", exist_ok=True)
        
        cna_sim = CNA_sim(num_sim=num_sim, seed=seed)
        cna_sim.save(output_dir)

        cna_sim.plot_realization_true_flat(f"{output_dir}/cna_sim_{num_sim}/plots/truth_rdr_baf_flat_{num_sim}.pdf")
        cna_sim.plot_realization_true_genome(f"{output_dir}/cna_sim_{num_sim}/plots/truth_rdr_baf_genome_{num_sim}.pdf")

    logger.info(f"\n\nDone ({time.time() - start:.3f} seconds).\n\n")


def main():
    # NB python python/cna_mixture/scripts/run_sim.py --output-dir ~/scratch/cna_mixture/sims/ --num_sims 2
    parser = argparse.ArgumentParser(description="Create CNA simulation.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        required=True,
        help="Directory to save the simulation results.",
    )
    parser.add_argument(
        "--num_sims",
        type=int,
        default=1,
        help="Number of simulations to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314,
        help="Seed for random number generation.",
    )

    args = parser.parse_args()

    run_sim(args.output_dir, args.num_sims, args.seed)


if __name__ == "__main__":
    main()
