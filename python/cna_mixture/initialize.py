import logging

import numpy as np

from cna_mixture.cna_emission import CNA_emission
from cna_mixture.plotting import plot_rdr_baf_flat
from cna_mixture.utils import deprecated

logger = logging.getLogger(__name__)


class CNA_mixture_initialize:
    def __init__(
        self,
        data,
        mixture_params,
        seed=314,
        mode="random",
    ):
        self.data = data
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.mode = mode
        self.params = mixture_params

    @property
    def rdr(self):
        # NB baseline coverage == Tn * lambdas, where lambdas.sum() == 1.
        return self.data["read_coverage"] / self.data["baseline_coverage"]

    @property
    def baf(self):
        return self.data["b_reads"] / self.data["snp_coverage"]

    @property
    def rdr_baf(self):
        return np.c_[self.rdr, self.baf]

    def run(self):
        match self.mode:
            case "random":
                mixture_params, cost = self.random()
                
            case "non_normal":
                mixture_params, cost = self.non_normal()
                
            case "plusplus":
                mixture_params, cost = self.plusplus()
            case _:
                msg = f"{self.mode} style initialization is not supported."
                raise ValueError(msg)

        return mixture_params, cost

    def random(self):
        # NB list of (baf, rdr) for k=4 states, without replacement.
        integers = self.rng.choice(
            np.arange(3, 10), size=self.params.num_cna_states, replace=False
        )

        # NB assumes single unit of quantized baf for given read depth.
        cna_states = [
            [1.0 * int_sample, 1.0 / int_sample] for int_sample in np.sort(integers)
        ]

        self.params.cna_states = np.array(
            [self.params.normal_state.tolist(), *cna_states]
        )

        self.params.verify()

        return self.params, np.inf

    def non_normal(self, threshold=0.05, non_normal=True):
        """
        Given an instance of (RDR, BAF) data, update the mixture params
        to be a random sample of the *non-normal* data, i.e. a copy number
        that is not unity.
        """
        rdr_baf = self.rdr_baf
        
        if non_normal:
            samples = rdr_baf[np.abs(rdr_baf[:, 0] - 1.0) > threshold]
        else:
            samples = rdr_baf.copy()

        logger.info(
            f"Initializing CNA mixture params with random_rdr_baf with non_normal={non_normal}"
        )

        xx = np.arange(len(samples))
        idx = self.rng.choice(xx, size=self.params.num_states - 1, replace=False)

        cna_states = np.vstack([self.params.normal_state, samples[idx]])

        self.params.cna_states = cna_states[cna_states[:, 0].argsort()]

        # TODO return cost.
        return self.params, np.inf

    # TODO provided with an emission model directly.
    @staticmethod
    def mixture_plusplus_cost(
        samples, centers, overdisp_phi, overdisp_tau, reduction=True
    ):
        cost = get_ln_state_emission(
            samples[:, 0],
            samples[:, 1],
            samples[:, 2],
            centers[:, 0],
            overdisp_phi,
            centers[:, 1],
            overdisp_tau,
        )

        if reduction:
            # NB emission probability for "most likely" state.
            cost = np.max(cost, axis=1)

        return -cost

    @deprecated
    def plusplus(self, ks, xs, bs, ns, N=4, validate=False):
        """
        Initialize with a mixture++ pattern, where subsequent selections are
        proportional to the cost for the current subset of states.
        """
        logger.info(f"Initializing CNA mixture params with {N}-greedy CNA_mixture++")

        ks = self.data["read_coverage"],
        xs = self.data["baseline_coverage"]
        bs = self.data["b_reads"],
        ns = self.data["snp_coverage"],

        # TODO
        em = CNA_emission(N, ks, xs, bs, ns, ws=None, backend=None, compress=True)
        
        idx = np.arange(len(ks))
        samples = np.c_[ks, xs, ns]

        # NB we assume a normal-like state to start.
        normal = self.normal_state.tolist()
        normal[0] *= self.genome_coverage

        centers = np.array([normal])

        cost = self.mixture_plusplus_cost(
            samples, centers, self.overdisp_phi, self.overdisp_tau
        )

        # NB one cost for normal state per sample.
        assert len(cost) == len(ks)

        logger.info(
            f"Initialized mixture++ with mixture++ cost for a normal state: {cost.sum()}"
        )

        while len(centers) < self.num_states:
            ps = cost / cost.sum()

            # TODO HACK
            if validate:
                tmp_cost = self.mixture_plusplus_cost(
                    samples,
                    centers,
                    self.overdisp_phi,
                    self.overdisp_tau,
                    collapse=True,
                )

                tmp_cost /= tmp_cost.max()

                states_bag = centers.copy()
                states_bag[:, 0] /= self.genome_coverage

                plot_rdr_baf_flat(
                    f"plots/mixture++_{len(centers)}_rdr_baf_flat.pdf",
                    ks / self.genome_coverage,
                    xs / ns,
                    ln_state_posteriors=np.log(tmp_cost),
                    states_bag=states_bag,
                    title=None,
                )

            new_samples = samples[self.rng.choice(idx, p=ps, size=N, replace=False)]

            # NB state read depth (RDR x genome coverage) and BAF.
            #
            # TODO here, we would also select based on BAF error, i.e. for high coverage.
            trial_centers = np.c_[
                new_samples[:, 0], new_samples[:, 1] / new_samples[:, 2]
            ]

            logger.debug(f"Found trial centers:\n{trial_centers}")

            costs = [
                self.mixture_plusplus_cost(
                    samples,
                    np.vstack([centers, tc]),
                    self.overdisp_phi,
                    self.overdisp_tau,
                )
                for tc in trial_centers
            ]

            costs_sum = np.array([cost.sum() for cost in costs])
            minimizer = np.argmin(costs_sum)

            cost = costs[minimizer]
            centers = np.vstack([centers, trial_centers[minimizer]])

        centers[:, 0] /= self.genome_coverage

        self.cna_states = centers.copy()
        self.cna_states = self.cna_states[self.cna_states[:, 0].argsort()]

        return self.cna_states, cost
