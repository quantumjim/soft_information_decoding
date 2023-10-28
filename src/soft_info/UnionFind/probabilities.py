# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-17

import warnings
from collections import defaultdict
from math import exp

import numpy as np
from qiskit.result import Counts

from Scratch import metadata_loader
from soft_info import get_repcode_IQ_map


def estimate_outcome(IQ_point, kernel_0=None, kernel_1=None):
    """Estimate the outcome for a given IQ datapoint.

    Parameters:
    - IQ_point: The IQ datapoint for which the outcome should be estimated.
    - kernel_0: KDE model for state 0.
    - kernel_1: KDE model for state 1.

    Returns:
    - int: The estimated outcome (0 or 1).
    """
    if kernel_0 is None or kernel_1 is None:
        warnings.warn(
            "Not enough kernels provided. Using the magnitude of the real part for estimation..")
        if np.real(IQ_point) > 0:
            return 1
        else:
            return 0

    if kernel_0.score_samples([IQ_point]) > kernel_1.score_samples([IQ_point]):
        return 0
    else:
        return 1


def get_counts(IQ_data, kde_dict=None, layout=None, synd_rounds=None):
    """Convert the IQ data to counts using corresponding KDEs for each qubit.

    Args:
        IQ_data (list of list of floats): The IQ data for multiple shots. 
            Each inner list contains IQ data for a single shot.
        kde_dict (dict, optional): Dictionary mapping qubit index to a tuple containing its KDEs for state 0 and 1.
        layout (list of int, optional): List specifying the layout of qubits.
        synd_rounds (int, optional): Number of syndrome rounds.

    Returns:
        count_dict (qiskit.result.counts.Counts): The counts for the experiments.
    """
    count_dict = defaultdict(int)

    qubit_mapping = None
    if layout is not None and synd_rounds is not None:
        qubit_mapping = get_repcode_IQ_map(layout, synd_rounds)

    if qubit_mapping is None:
        warnings.warn(
            "Missing layout or synd_rounds, estimating outcomes without KDEs.")

    for shot in IQ_data:
        outcome_str = ""
        for idx, IQ_point in enumerate(shot):
            if qubit_mapping is not None:
                qubit_idx = qubit_mapping[idx]
                kde_0, kde_1 = kde_dict.get(qubit_idx, (None, None))
                # returns None if qubit_idx not in dict => normal outcome estimation
                outcome_str += str(estimate_outcome(IQ_point, kde_0, kde_1))
            else:
                outcome_str += str(estimate_outcome(IQ_point))
        count_dict[outcome_str] += 1

    return Counts(count_dict)


# NOT est_outcome left because will compute it already for the graph
def llh_ratio(IQ_point, kernel_0, kernel_1):
    """Compute the likelihood ratio for a given IQ_point according to arXiv:2107.13589.

    Args:
        IQ_point (float): The IQ datapoint.
        kernel_0: KDE model for state 0.
        kernel_1: KDE model for state 1.

    Returns:
        float: The likelihood ratio.
    """

    prob_0 = exp(kernel_0.score_samples([IQ_point])[0])
    prob_1 = exp(kernel_1.score_samples([IQ_point])[0])

    est_outcome = estimate_outcome(IQ_point, kernel_0, kernel_1)

    if est_outcome in [0, 1]:
        return prob_1 / prob_0 if est_outcome == 0 else prob_0 / prob_1
    else:
        raise ValueError("The estimated outcome must be either 0 or 1.")
