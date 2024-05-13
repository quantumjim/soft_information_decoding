from typing import Tuple
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from .gmm_fitting import fit_gmm_0_calib, process_gmm_data, get_gmm_RepCodeData, plot_RepCode_gmm
from sklearn.preprocessing import StandardScaler


def GMMIQConvertor(IQ_data: np.ndarray, 
                   all_memories: dict, 
                   inverted_q_map: dict, 
                   plot: bool = False) -> Tuple[Tuple[np.ndarray], dict]:
    """
    This function takes in IQ data and converts it to a GMM representation. It uses the calibration data from the memories to fit the GMMs.
    The function returns the GMM representations of the IQ data and the GMMs used to convert the data.

    Parameters:
       - IQ_data (np.ndarray): The IQ data to be converted to GMM representation.
       - all_memories (dict): The calibration data from the memories.
       - inverted_q_map (dict): The mapping from physical qubits to the IQ data.
       - plot (bool): Whether to plot the GMM representations or not.

    Returns:
       - Tuple[Tuple[np.ndarray], dict]: (countMat, pSoft, estim0matrix, estim1matrix, estim2matrix), gmm_dict
                                    The GMM representations of the IQ data and the GMMs used to convert the data.
                                    
    """
    countMat = np.zeros_like(IQ_data, dtype=int)
    pSoft, estim0matrix, estim1matrix, estim2matrix = (np.zeros_like(IQ_data, dtype=float) for _ in range(4))

    gmm_dict = {}
    for phys_idx, col_indices in tqdm(inverted_q_map.items()):
        IQ_data_cols = IQ_data[:, col_indices].flatten()
        mmr_0 = all_memories[phys_idx]['mmr_0']

        IQ_data_proc, mmr_0_proc, _ = process_gmm_data(IQ_data_cols, mmr_0)
        gmm_0 = fit_gmm_0_calib(mmr_0_proc)
        gmm = get_gmm_RepCodeData(IQ_data_proc, gmm_0)
        gmm_dict[phys_idx] = gmm
        plot = plot_RepCode_gmm(IQ_data_proc, gmm) if plot else None

        probas = gmm.predict_proba(IQ_data_proc) + 1e-30

        # reorder to make sure 0, 1, 2 corresponds to states
        probas = reorder_gmm_components(gmm, probas, phys_idx)

        classifications = np.argmax(probas, axis=1)
        countMat[:, col_indices] = classifications.reshape(-1, len(col_indices))

        pSoft[:, col_indices] = (1 / (1 + np.max(probas[:, :2], axis=1) / np.min(probas[:, :2], axis=1))).reshape(-1, len(col_indices))   
        # pSoft[:, col_indices] = (1 / (1 + np.max(probas, axis=1) / np.min(probas, axis=1))).reshape(-1, len(col_indices))   

        estim0matrix[:, col_indices] = probas[:, 0].reshape(-1, len(col_indices))  
        estim1matrix[:, col_indices] = probas[:, 1].reshape(-1, len(col_indices)) if probas.shape[1] > 1 else None
        estim2matrix[:, col_indices] = probas[:, 2].reshape(-1, len(col_indices)) if probas.shape[1] > 2 else None
        

    return (countMat, pSoft, estim0matrix, estim1matrix, estim2matrix), gmm_dict

def reorder_gmm_components(gmm, probas, qubit):
    # Ensure Probas 0 corresponds to the component with the smallest x-value in its mean
    smallest_x_index = np.argmin(gmm.means_[:, 0])
    if smallest_x_index != 0:
        print(f"Reordering qubit {qubit}: Component with smallest x-value is not at index 0, but at index {smallest_x_index}.")
        new_order = np.arange(gmm.n_components)
        new_order[0], new_order[smallest_x_index] = new_order[smallest_x_index], new_order[0]
        probas = probas[:, new_order]

    # Check and reorder to ensure that for the two other modes the one with the lowest weight is 2
    if gmm.n_components > 2:
        if gmm.weights_[1] > gmm.weights_[2]:
            print(f"Reordering qubit {qubit}: Component at index 2 does not have the lowest weight.")
            probas[:, [1, 2]] = probas[:, [2, 1]]
    
    return probas
            



def gaussianIQConvertor(IQ_data: np.ndarray, inverted_q_map: dict, gmm_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    nb_shots = IQ_data.shape[0]
    # Initialize matrices with the correct dimensions
    countMat = np.zeros((nb_shots, IQ_data.shape[1]), dtype=int)
    pSoft = np.zeros((nb_shots, IQ_data.shape[1]), dtype=float)

    for phys_idx, col_indices in inverted_q_map.items():
        gmm = gmm_dict[phys_idx]['gmm']
        scaler = gmm_dict[phys_idx]['scaler']

        if gmm.means_[0] > gmm.means_[1]:
            gmm.means_ = gmm.means_[::-1]
            gmm.weights_ = gmm.weights_[::-1]
            print("Warning: GMM means were inverted to match the expected order of the classes (0, 1)")

        for col_idx in col_indices: # slower but better apparently
            iq_real_data = IQ_data[:, col_idx].real.reshape(-1, 1)
            iq_data_scaled = scaler.transform(iq_real_data)
            probas = gmm.predict_proba(iq_data_scaled)

            class_labels = (probas[:, 1] > probas[:, 0]).astype(int)
            pSoftValues = 1 / (1 + np.max(probas, axis=1) / np.min(probas, axis=1))
            countMat[:, col_idx] = class_labels
            pSoft[:, col_idx] = pSoftValues

    return countMat, pSoft




def gaussianKDEIQconvertor(IQ_data: np.ndarray, inverted_q_map: dict, kde_dict: dict, scaler_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    countMat = np.zeros(IQ_data.shape, dtype=int)
    pSoft = np.zeros(IQ_data.shape, dtype=float)

    for phys_idx, col_indices in tqdm(inverted_q_map.items()):
        kde_0, kde_1 = kde_dict[phys_idx]
        scaler = scaler_dict[phys_idx]

        for col_idx in col_indices:
            data = IQ_data[:, col_idx].flatten()
            combined_data = np.column_stack((data.real, data.imag))
            norm_data = scaler.transform(combined_data)

            log_prob_0 = kde_0.score_samples(norm_data)
            log_prob_1 = kde_1.score_samples(norm_data)
            
            class_labels = (log_prob_1 > log_prob_0).astype(int)

            prob_0 = np.exp(log_prob_0)
            prob_1 = np.exp(log_prob_1)

            pSoftValues = 1 / (1 + np.max(np.column_stack((prob_0, prob_1)), axis=1) / (np.min(np.column_stack((prob_0, prob_1)), axis=1)+1e-12))

            countMat[:, col_idx] = class_labels
            pSoft[:, col_idx] = pSoftValues

    return countMat, pSoft




# def gaussianIQConvertor(IQ_data: np.ndarray, inverted_q_map: dict, gmm_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
#     nb_shots = IQ_data.shape[0]
#     countMat = np.zeros_like(IQ_data.real, dtype=int)
#     pSoft = np.zeros_like(IQ_data.real, dtype=float)  # Initialize the misassignment matrix

#     for qubit_idx, phys_indices in inverted_q_map.items():
#         gmm = gmm_dict[qubit_idx]['gmm']
#         scaler = gmm_dict[qubit_idx]['scaler']

#         iq_subset_scaled = scaler.transform(IQ_data[:, phys_indices].real.reshape(-1, 1))
#         probas = gmm.predict_proba(iq_subset_scaled)

#         class_1_greater = probas[:, 1] > probas[:, 0]
#         class_labels = class_1_greater.astype(int)

#         # Calculate misassignment probabilities (pSoft)
#         pSoftValues = 1 / (1 + np.max(probas, axis=1) / np.min(probas, axis=1))

#         # Assign the computed labels and misassignment probabilities into the matrices
#         for i, phys_idx in enumerate(phys_indices):
#             countMat[:, phys_idx] = class_labels[i*nb_shots:(i+1)*nb_shots]
#             pSoft[:, phys_idx] = pSoftValues[i*nb_shots:(i+1)*nb_shots]

#     return countMat, pSoft
