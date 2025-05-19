import logging

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from cna_mixture.utils import patch_default, tophat_smooth

logger = logging.getLogger(__name__)

def sample_colormap(num_states, cmap="tab20b"):
    cmap = plt.get_cmap(cmap)    
    indices = np.linspace(0, 1, num_states)
        
    return [cmap(i) for i in indices]


def ln_probs_to_rgb(ln_probs):
    nrows, ncols = ln_probs.shape

    rgb = np.zeros(shape=(len(ln_probs), 3))
    alpha, cmap = 0.25, None
    
    if ncols == 1:
        alpha = np.exp(ln_probs)

    elif ncols > 5:
        colors = sample_colormap(ncols)
        
        idx = np.argmax(ln_probs, axis=1)
        rgb = [colors[ii] for ii in idx]
        cmap = None

    # TODO
    else:
        rgb = np.zeros(shape=(len(ln_probs), 3))
        alpha = 0.25

        for ii in range(ln_probs.shape[1]):
            if ii <= 2:
                rgb[:, -(1 + ii)] = np.exp(ln_probs[:, -(1 + ii)])
            else:
                logger.warning(
                    f"Failed to map all of {ln_probs.shape[1]} states to RGB when plotting"
                )
                break

    return rgb, alpha, cmap


def plot_rdr_baf_flat(
    fpath,
    rdr,
    baf,
    ln_state_posteriors=None,
    states_bag=None,
    title=None,
):
    """
    NB state_posteriors may be an integer, corresponding to a decoded state, or
       the posterior probs. for up to four states, which are mapped to RGB +
       alpha transparency.
    """
    pl.clf()

    assert len(ln_state_posteriors) == len(
        rdr
    ), f"Found inconsistent RDR, BAF and state posteriors (size {len(rdr)} and {len(ln_state_posteriors)} respectively)"

    rgb, alpha, cmap = ln_probs_to_rgb(ln_state_posteriors)

    pl.axhline(0.5, c="k", lw=0.5)
    plt.scatter(rdr, baf, c=rgb, marker=".", lw=0.0, alpha=alpha, cmap=cmap)

    if states_bag is not None:
        for state_rdr, state_baf in states_bag:
            pl.scatter(
                state_rdr,
                state_baf,
                marker="*",
                edgecolors="black",
                facecolors="white",
                s=45,
            )

    pl.xlim(-0.05, 15.0)
    pl.ylim(-0.05, 1.05)

    pl.xlabel(r"$\mu_{\rm RDR}$")
    pl.ylabel(r"$p_{\rm BAF}$")

    if title is not None:
        pl.title(title)

    pl.savefig(fpath)

    logger.info(f"Plotted rdr_baf_flat to {fpath}")


def plot_rdr_baf_genome(
    fpath,
    rdr,
    baf,
    ln_state_posteriors=None,
    states_bag=None,
    title=None,
    outliers_mask=None,
):
    pl.clf()

    assert len(ln_state_posteriors) == len(
        rdr
    ), f"Found inconsistent RDR, BAF and state posteriors (size {len(rdr)} and {len(ln_state_posteriors)} respectively)"
    
    segment_index = np.arange(len(rdr))

    figsize = (15, 10)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)

    for state_rdr, state_baf in states_bag:
        axes[0].axhline(state_rdr, c="k", lw=0.1)
        axes[1].axhline(state_baf, c="k", lw=0.1)

    rgb, alpha, cmap = ln_probs_to_rgb(ln_state_posteriors)

    axes[0].scatter(
        segment_index, rdr, c=rgb, marker=".", lw=0.0, alpha=alpha, cmap=cmap
    )

    axes[1].scatter(
        segment_index, baf, c=rgb, marker=".", lw=0.0, alpha=alpha, cmap=cmap
    )

    valid = np.isfinite(rdr)

    axes[0].set_xlim(-100, len(rdr))
    axes[0].set_ylim(-0.5, np.percentile(rdr[valid], 99.0))
    axes[0].set_ylabel(r"read depth ratio")

    axes[1].set_ylabel(r"$b$-allele frequency")
    axes[1].set_xlabel("segment index")

    if title is not None:
        pl.title(title)

    pl.savefig(fpath)

    logger.info(f"Plotted rdr_baf_genome to {fpath}")
