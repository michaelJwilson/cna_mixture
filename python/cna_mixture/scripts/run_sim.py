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


def run_sim(output_dir, plots_dir):
    start = time.time()

    # NB ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    cna_sim = CNA_sim()
    cna_sim.save(output_dir)

    # cna_sim.plot_realization_true_flat(f"{plots_dir}/truth_rdr_baf_flat.pdf")
    # cna_sim.plot_realization_true_genome(f"{plots_dir}/truth_rdr_baf_genome.pdf")

    logger.info(f"\n\nDone ({time.time() - start:.3f} seconds).\n\n")


def main():
    parser = argparse.ArgumentParser(description="Create CNA simulation.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        required=True,
        help="Directory to save the simulation results.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="./plots",
        help="Directory to save the validation plots.",
    )

    args = parser.parse_args()

    run_sim(args.output_dir, args.plots_dir)


if __name__ == "__main__":
    main()
