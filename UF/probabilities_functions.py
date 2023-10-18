# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-17

import numpy as np
from collections import defaultdict

from qiskit.result import Counts


# TODO sample + fitting procedure to get the distributions
def get_distr(device, qubit):
    """Fit the IQ Distribution for a given qubit in a given device.

    Args:
        device

    """
    # TODO implement this
    distr_0 = None
    distr_1 = None

    return distr_0, distr_1


def estimate_outcome(IQ_point, distr_0=None, distr_1=None):
    """Estimate the outcome for a given IQ datapoint. If 
    no distribution given uses the real part of the IQ point."""
    if distr_0 is None or distr_1 is None:
        if np.real(IQ_point) > 0:
            return 1
        else: # more probable to get a 0 if it is in the middle
            return 0
        
    if distr_0(IQ_point) > distr_1(IQ_point):
        return 0
    else:
        return 1


def get_counts(IQ_data, distr_0=None, distr_1=None):
    """Convert the IQ data to counts.

    Args:
        IQ_data (dict): The IQ data for multiple shots.

    Returns:
        count dict (qiskit.result.counts.Counts): The counts for the experiments.
    """
    count_dict = defaultdict(int)
    for shot in IQ_data:
        outcome_str = ""
        for IQ_point in shot:
            outcome_str += str(estimate_outcome(IQ_point, distr_0, distr_1))
        count_dict[outcome_str] += 1

    return Counts(count_dict)


# NOT est_outcome left because will compute it already for the graph
def llh_ratio(IQ_point, distr_0, distr_1):
    """Compute the likelihood ratio for a given mu and mu_hat. According to arXiv:2107.13589.

    Args:
        mu (float): The IQ datapoint.
        mu_hat (float): The estimated outcome.
        dist_0 (scipy.stats distribution): The IQ distribution for outcome = 0.
        dist_1 (scipy.stats distribution): The IQ distribution for outcome = 1.

    Returns:
        float: The log-likelihood ratio.
    """
    probabilities = {0: distr_0(IQ_point), 1: distr_1(IQ_point)}
    est_outcome = estimate_outcome(IQ_point, distr_0, distr_1)
    if est_outcome in [0, 1]:
        return probabilities[1 - est_outcome] / probabilities[est_outcome]
    else:
        raise ValueError("The estimated outcome must be either 0 or 1.")
