# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-17

import warnings
from collections import defaultdict
from math import exp

import numpy as np
from tqdm import tqdm
from qiskit.result import Counts

from ..Hardware.transpile_rep_code import get_repcode_IQ_map


def estimate_outcome(IQ_point, kernel_0=None, kernel_1=None, scaler=None):
    """Estimate the outcome for a given IQ datapoint.

    Parameters:
    - IQ_point: The IQ datapoint for which the outcome should be estimated.
    - kernel_0: KDE model for state 0.
    - kernel_1: KDE model for state 1.

    Returns:
    - int: The estimated outcome (0 or 1).
    """
    if kernel_0 is None or kernel_1 is None or scaler is None:
        warnings.warn(
            "Not enough kernels or no scaler provided. Using the magnitude of the real part for estimation..")
        if np.real(IQ_point) > 0:
            return 1
        else:
            return 0
    
    scaled_plane_point = scaler.transform([[np.real(IQ_point), np.imag(IQ_point)]])

    if kernel_0.score_samples(scaled_plane_point) > kernel_1.score_samples(scaled_plane_point):
        return 0
    else:
        return 1


def get_counts(IQ_data, kde_dict=None, scaler_dict=None, layout=None, synd_rounds=None):
    """Convert the IQ data to counts using corresponding KDEs for each qubit.

    Args:
        IQ_data (list of list of floats): The IQ data for multiple shots. 
            Each inner list contains IQ data for a single shot.
        kde_dict (dict, optional): Dictionary mapping qubit index to a tuple containing its KDEs for state 0 and 1.
        scaler_dict (dict, optional): Dictionary mapping qubit index to its scaler for normalization.
        layout (list of int, optional): List specifying the layout of qubits. [link_qubits, code_qubits]
        synd_rounds (int, optional): Number of syndrome rounds.

    Returns:
        count_dict (qiskit.result.counts.Counts): The counts for the experiments. Ordered by number of shots.
    """
    count_dict = defaultdict(int)

    qubit_mapping = None
    if layout is not None and synd_rounds is not None:
        qubit_mapping = get_repcode_IQ_map(layout, synd_rounds)

    if qubit_mapping is None:
        warnings.warn(
            "Missing layout or synd_rounds, estimating outcomes without KDEs.")

    for shot in tqdm(IQ_data, desc=f"Processing {len(IQ_data)} shots"):
        outcome_str = ""
        for idx, IQ_point in enumerate(shot):
            if qubit_mapping is not None:
                qubit_idx = qubit_mapping[idx]
                kde_0, kde_1 = kde_dict.get(qubit_idx, (None, None))
                scaler = scaler_dict.get(qubit_idx, None)
                # returns None if qubit_idx not in dict => normal outcome estimation
                outcome_str += str(estimate_outcome(IQ_point, kde_0, kde_1, scaler))
            else:
                outcome_str += str(estimate_outcome(IQ_point))
        count_dict[outcome_str] += 1

    count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))

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
