import logging
import time

import numpy as np
from cna_mixture.cna_inference import CNA_inference
from cna_mixture.cna_sim import CNA_sim
from cna_mixture.plotting import plot_rdr_baf_genome

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


def main():
    start = time.time()

    cna_sim = CNA_sim()
    cna_sim.plot_realization_true_flat("plots/truth_rdr_baf_flat.pdf")
    cna_sim.plot_realization_true_genome("plots/truth_rdr_baf_genome.pdf")

    # fit_gaussian_mixture(cna_sim.rdr_baf)

    # NB total number of states (inc. normal).
    # res = CNA_inference(cna_sim.num_states, cna_sim.genome_coverage, cna_sim.data).fit()

    logger.info(f"\n\nDone ({time.time() - start:.3f} seconds).\n\n")


if __name__ == "__main__":
    main()
