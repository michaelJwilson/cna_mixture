import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


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
        if ln_state_posteriors.ndim == 1:
            rgb = np.exp(ln_state_posteriors)
            alpha = 0.25
            cmap = "viridis"

        else:
            assert ln_state_posteriors.shape[1] == 4

            # NB assumed to be normal probability.
            # alpha = 0.25 + 3.0 * (1.0 - state_posteriors[:, 0]) / 4.0
            alpha = 0.25

            rgb = np.exp(ln_state_posteriors[:, 1:4])
            cmap = None

    pl.axhline(0.5, c="k", lw=0.5)
    plt.scatter(rdr, baf, c=rgb, marker=".", lw=0.0, alpha=alpha, cmap=cmap)

    if states_bag is not None:
        for rdr, baf in states_bag:
            pl.scatter(
                rdr, baf, marker="*", edgecolors="black", facecolors="white", s=45
            )

    pl.xlim(-0.05, 15.0)
    pl.ylim(-0.05, 1.05)

    pl.xlabel(r"$\mu_{\rm RDR}$")
    pl.ylabel(r"$p_{\rm BAF}$")

    if title is not None:
        pl.title(title)

    pl.savefig(fpath)


def tophat_smooth(data, window_size):
    """
    Top-hat convolution of a 1D signal.
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def plot_rdr_baf_genome(
    fpath, rdr_baf, ln_state_posteriors=None, states_bag=None, title=None
):
    pl.clf()
    
    segment_index = np.arange(len(rdr_baf))

    figsize = (15, 10)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)

    rdr, baf = rdr_baf[:, 0], rdr_baf[:, 1]

    smooth_rdr = tophat_smooth(rdr, window_size=100)
    smooth_baf = tophat_smooth(baf, window_size=100)

    axes[0].plot(segment_index, rdr)
    axes[0].plot(segment_index, smooth_rdr)

    axes[1].plot(segment_index, baf)
    axes[1].plot(segment_index, smooth_baf)

    axes[0].set_ylabel(r"read depth ratio")

    axes[1].set_ylabel(r"$b$-allele frequency")
    axes[1].set_xlabel("intervals")

    pl.savefig(fpath)
