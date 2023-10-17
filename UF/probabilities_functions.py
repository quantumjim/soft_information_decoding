# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-17

import numpy as np


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

def estimate_outcome(IQ_point, distr_0, distr_1):
    """Estimate the outcome for a given IQ datapoint."""
    if distr_0(IQ_point) > distr_1(IQ_point):
        return 0
    else:
        return 1
    
def get_counts(IQ_data):
    """Convert the IQ data to counts.
    
    Args:
        IQ_data (dict): The IQ data for multiple shots.

    Returns:
        count dict (qiskit.result.counts.Counts): The counts for the experiments.
    """
    # TODO implement this
    return 0


def llh_ratio(IQ_point, distr_0, distr_1): # NOT est_outcome left because will compute it already for the graph
    """Compute the log-likelihood ratio for a given mu and mu_hat. According to arXiv:2107.13589.

    Args:
        mu (float): The IQ datapoint.
        mu_hat (float): The estimated outcome.
        dist_0 (scipy.stats distribution): The IQ distribution for outcome = 0.
        dist_1 (scipy.stats distribution): The IQ distribution for outcome = 1.

    Returns:
        float: The log-likelihood ratio.
    """
    log_dists = {0: np.log(distr_0(IQ_point)), 1: np.log(distr_1(IQ_point))}

    est_outcome = estimate_outcome(IQ_point, distr_0, distr_1)
    if est_outcome in [0, 1]:
        return log_dists[1 - est_outcome] - log_dists[est_outcome]
    else:
        raise ValueError("The estimated outcome must be either 0 or 1.")
    


