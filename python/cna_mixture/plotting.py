import logging

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

logger = logging.getLogger(__name__)

def ln_probs_to_rgb(ln_probs):
    if ln_probs.ndim == 1:
        # NB black
        rgb = np.zeros(shape=(len(ln_probs), 3))
        alpha = np.exp(ln_probs)
        
    else:
        # NB assumed to be normal probability.
        rgb = np.zeros(shape=(len(ln_probs), 3))
        alpha = 0.25
        
        for ii in range(ln_probs.shape[1]):
            if ii <= 2:
                rgb[:, -(1 + ii)] = np.exp(ln_probs[:, -(1 + ii)])
            else:
                logger.warning(f"Failed to map all of {ln_probs.shape[1]} states to RGB when plotting")
                break
                
    cmap = None
        
    return rgb, alpha, cmap


def plot_rdr_baf_flat(
    fpath, rdr, baf, ln_state_posteriors=None, states_bag=None, title=None
):
    """
    NB state_posteriors may be an integer, corresponding to a decoded state, or
       the posterior probs. for up to four states, which are mapped to RGB +
       alpha transparency.
    """
    pl.clf()

    if ln_state_posteriors is not None:
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


def tophat_smooth(data, window_size):
    """
    Top-hat convolution of a 1D signal.
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def plot_rdr_baf_genome(
    fpath, rdr, baf, ln_state_posteriors=None, states_bag=None, title=None
):
    pl.clf()

    segment_index = np.arange(len(rdr))

    figsize = (15, 10)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)

    for state_rdr, state_baf in states_bag:
        axes[0].axhline(state_rdr, c="k", lw=0.1)
        axes[1].axhline(state_baf, c="k", lw=0.1)

    # smooth_rdr = tophat_smooth(rdr, window_size=100)
    # smooth_baf = tophat_smooth(baf, window_size=100)

    rgb, alpha, cmap = ln_probs_to_rgb(ln_state_posteriors)

    axes[0].set_xlim(-100, 10_100)

    axes[0].scatter(
        segment_index, rdr, c=rgb, marker=".", lw=0.0, alpha=alpha, cmap=cmap
    )
    # axes[0].scatter(segment_index, smooth_rdr)

    axes[1].scatter(
        segment_index, baf, c=rgb, marker=".", lw=0.0, alpha=alpha, cmap=cmap
    )
    # axes[1].plot(segment_index, baf)
    # axes[1].plot(segment_index, smooth_baf)

    axes[0].set_ylabel(r"read depth ratio")

    axes[1].set_ylabel(r"$b$-allele frequency")
    axes[1].set_xlabel("segment index")

    if title is not None:
        pl.title(title)

    pl.savefig(fpath)

    logger.info(f"Plotted rdr_baf_genome to {fpath}")
