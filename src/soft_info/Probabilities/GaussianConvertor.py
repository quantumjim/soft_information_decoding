from typing import Tuple

import numpy as np

def gaussianIQConvertor(IQ_data: np.ndarray, inverted_q_map: dict, gmm_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    nb_shots = IQ_data.shape[0]
    countMat = np.zeros_like(IQ_data.real, dtype=int)
    pSoft = np.zeros_like(IQ_data.real, dtype=float)  # Initialize the misassignment matrix

    for qubit_idx, phys_indices in inverted_q_map.items():
        gmm = gmm_dict[qubit_idx]['gmm']
        scaler = gmm_dict[qubit_idx]['scaler']

        iq_subset_scaled = scaler.transform(IQ_data[:, phys_indices].real.reshape(-1, 1))
        probas = gmm.predict_proba(iq_subset_scaled)

        class_1_greater = probas[:, 1] > probas[:, 0]
        class_labels = class_1_greater.astype(int)

        # Calculate misassignment probabilities (pSoft)
        pSoft = 1 / (1 + np.max(probas, axis=1) / np.min(probas, axis=1))

        # Assign the computed labels and misassignment probabilities into the matrices
        for i, phys_idx in enumerate(phys_indices):
            countMat[:, phys_idx] = class_labels[i*nb_shots:(i+1)*nb_shots]
            pSoft[:, phys_idx] = pSoft[i*nb_shots:(i+1)*nb_shots]

    return countMat, pSoft
