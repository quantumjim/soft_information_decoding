from typing import Tuple
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm





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
