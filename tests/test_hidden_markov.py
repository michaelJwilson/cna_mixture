from __future__ import annotations
import numpy as np
import numpy.testing as npt
from cna_mixture.state_priors import CNA_categorical_prior, CNA_markov_prior
from scipy.special import logsumexp
from scipy.stats import norm

