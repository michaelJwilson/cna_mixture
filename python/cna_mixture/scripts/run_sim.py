import argparse
import logging
import time
from pathlib import Path

from cna_mixture.cna_sim import CNA_sim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def run_sim(output_dir, num_sims=1, seed=314):
    start = time.time()
    
    for sim_id in range(num_sims):
        # NB ensure the directory exists
        Path.mkdir(f"{output_dir}/cna_sim_{sim_id}/plots", exist_ok=True, parents=True)
        
        cna_sim = CNA_sim(sim_id=sim_id, seed=seed)
        cna_sim.save(output_dir)

        cna_sim.plot_realization_true_flat(f"{output_dir}/cna_sim_{sim_id}/plots/truth_rdr_baf_flat_{sim_id}.pdf")
        cna_sim.plot_realization_true_genome(f"{output_dir}/cna_sim_{sim_id}/plots/truth_rdr_baf_genome_{sim_id}.pdf")

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
