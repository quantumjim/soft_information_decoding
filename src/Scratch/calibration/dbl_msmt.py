from typing import Tuple

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np


def postselect_calib_data(qubit_data_dict: dict) -> Tuple[dict, dict]:
    """
    Processes and postselects qubit calibration data based on Gaussian Mixture Model (GMM) predictions,
    excluding instances where both initial and final measurements are incorrect. The function fits a GMM
    to rescaled measurement data, predicts labels for both initial and second measurements, and retains
    data except where both measurements are classified as incorrect for each qubit state. Additionally,
    it calculates hard and soft measurement error probabilities for each qubit based on the GMM predictions.

    Parameters:
    - qubit_data_dict (dict): Dictionary with qubit indices as keys and values being another dictionary
      containing arrays of initial and second measurements ('mmr_0', 'mmr_1', 'mmr_0_scnd', 'mmr_1_scnd').

    Returns:
    - tuple of (dict, dict):
        - First dict: Processed data for each qubit including postselected measurements ('mmr_0', 'mmr_1') and the
          original second measurements ('mmr_0_scnd', 'mmr_1_scnd'), along with the fitted GMM object and scaler.
          I.e., {qubit_idx: {'mmr_0': ..., 'mmr_1': ..., 'mmr_0_scnd': ..., 'mmr_1_scnd': ..., 'gmm': ..., 'scaler': ...}}
        
        - Second dict: Measurement error probabilities for each qubit, including 'p_hard' (probability of both
          measurements being incorrect) and 'p_soft' (probability of only one measurement being incorrect).
          I.e., {qubit_idx: {'p_hard': ..., 'p_soft': ...}}

    The method aims to improve measurement reliability by filtering out likely incorrect measurement
    outcomes, based on the assumption that errors in both initial and final measurements are less common.
    It further quantifies the measurement reliability through hard and soft error probabilities.
    """ 
    processed_data = {}
    msmt_err_probs = {}    
    for qubit_idx, data in qubit_data_dict.items():
        mmr_tot = np.concatenate([data['mmr_0'].real, data['mmr_1'].real, 
                                    data['mmr_0_scnd'].real, data['mmr_1_scnd'].real])
        
        # Rescale the data
        scaler = StandardScaler()
        mmr_tot_s = scaler.fit_transform(mmr_tot.reshape(-1, 1))
        
        # Fit the GMM
        gmm = GaussianMixture(n_components=2, covariance_type='tied', random_state=42)
        gmm.fit(mmr_tot_s)

        # Reorder the labels
        if gmm.means_[0] > gmm.means_[1]:
            gmm.means_ = gmm.means_[::-1]
            gmm.weights_ = gmm.weights_[::-1]
        
        # Predict labels for both initial and second measurements
        labels_0 = gmm.predict(scaler.transform(data['mmr_0'].real.reshape(-1, 1)))
        labels_1 = gmm.predict(scaler.transform(data['mmr_1'].real.reshape(-1, 1)))
        labels_0_scnd = gmm.predict(scaler.transform(data['mmr_0_scnd'].real.reshape(-1, 1)))
        labels_1_scnd = gmm.predict(scaler.transform(data['mmr_1_scnd'].real.reshape(-1, 1)))

        # Identify where either the initial or final measurement (or both) are correct
        # This excludes cases where both initial and final measurements are wrong
        correct_or_mixed_0 = ~((labels_0 == 1) & (labels_0_scnd == 1))
        correct_or_mixed_1 = ~((labels_1 == 0) & (labels_1_scnd == 0))

        # Apply postselection
        mmr_0 = data['mmr_0'][correct_or_mixed_0]
        mmr_1 = data['mmr_1'][correct_or_mixed_1]
        
        # Store processed data
        processed_data[qubit_idx] = {
            'mmr_0': mmr_0,
            'mmr_1': mmr_1,
            'mmr_0_scnd': data['mmr_0_scnd'],
            'mmr_1_scnd': data['mmr_1_scnd'],
            'gmm': gmm,
            'scaler': scaler,
        }

        # Get the error probabilities
        hard_prob_0 = ((labels_0 == 1) & (labels_0_scnd == 1)).sum() / len(labels_0)
        hard_prob_1 = ((labels_1 == 0) & (labels_1_scnd == 0)).sum() / len(labels_1)
        soft_prob_0 = ((labels_0 == 1) & (labels_0_scnd == 0)).sum() / len(labels_0)
        soft_prb_1 = ((labels_1 == 0) & (labels_1_scnd == 1)).sum() / len(labels_1)

        msmt_err_probs[qubit_idx] = {
            'p_hard': (hard_prob_0 + hard_prob_1) / 2,
            'p_soft': (soft_prob_0 + soft_prb_1) / 2
        }
             
    
    return processed_data, msmt_err_probs


def soft_postselect_calib_data(qubit_data_dict: dict, threshold: float):
    """
    Processes qubit calibration data by applying a Gaussian Mixture Model (GMM) for misassignment probability
    estimation and filters the data based on a specified threshold for soft postselection.

    Args:
    - qubit_data_dict (dict): A dictionary containing qubit calibration data. Expected to have qubit indices as keys
      and values being another dict with keys 'mmr_0', 'mmr_1', 'mmr_0_scnd', 'mmr_1_scnd', each mapping to an array
      of measurement results.
    - threshold (float): The threshold for filtering based on misassignment probabilities. Only data points with
      a misassignment probability below this threshold are retained.

    Returns:
    - dict: A dictionary containing the processed data for each qubit. For each qubit, the dictionary includes
      filtered measurement results ('mmr_0', 'mmr_1', 'mmr_0_scnd', 'mmr_1_scnd'),
      the fitted GMM object ('gmm'), and the scaler object ('scaler') used for data standardization.

    This function fits a GMM to the real parts of concatenated measurement results, calculates misassignment
    probabilities for second measurements, and filters the data based on the provided threshold. It aims to identify
    and retain only those data points that are less likely to be misassigned, according to the model's predictions.
    """
    processed_data = {}
    
    for qubit_idx, data in qubit_data_dict.items():
        mmr_tot = np.concatenate([data['mmr_0'].real, data['mmr_1'].real, 
                                    data['mmr_0_scnd'].real, data['mmr_1_scnd'].real])
        
        # Rescale the data
        scaler = StandardScaler()
        mmr_tot_s = scaler.fit_transform(mmr_tot.reshape(-1, 1))
        
        # Fit the GMM
        gmm = GaussianMixture(n_components=2, covariance_type='tied', random_state=42)
        gmm.fit(mmr_tot_s)

        # Reorder the labels
        if gmm.means_[0] > gmm.means_[1]:
            gmm.means_ = gmm.means_[::-1]
            gmm.weights_ = gmm.weights_[::-1]
        
        # Predict misassignment probabilities for second measurements
        pSoft_0_scnd = gmm.predict_proba(scaler.transform(data['mmr_0_scnd'].real.reshape(-1, 1)))
        pSoft_1_scnd = gmm.predict_proba(scaler.transform(data['mmr_1_scnd'].real.reshape(-1, 1)))
        
        # Calculate misassignment probabilities (pSoft)
        pSoft_0 = 1 / (1 + np.max(pSoft_0_scnd, axis=1) / np.min(pSoft_0_scnd, axis=1))
        pSoft_1 = 1 / (1 + np.max(pSoft_1_scnd, axis=1) / np.min(pSoft_1_scnd, axis=1))
        
        # Filter based on the threshold
        filter_mask_0 = pSoft_0 < threshold
        filter_mask_1 = pSoft_1 < threshold
        
        # Apply filter to original and second measurements
        mmr_0 = data['mmr_0'][filter_mask_0]
        mmr_1 = data['mmr_1'][filter_mask_1]
        mmr_0_scnd = data['mmr_0_scnd'][filter_mask_0]
        mmr_1_scnd = data['mmr_1_scnd'][filter_mask_1]
        
        # Store processed data
        processed_data[qubit_idx] = {
            'mmr_0': mmr_0,
            'mmr_1': mmr_1,
            'mmr_0_scnd': mmr_0_scnd,
            'mmr_1_scnd': mmr_1_scnd,
            'gmm': gmm,
            'scaler': scaler,
        }
    
    return processed_data


def postselect_exclude_double_wrong(processed_data: dict):
    """
    Postselects and retains data for mmr_0 and mmr_1 except where both the initial and final 
    measurements are incorrect, based on Gaussian Mixture Model (GMM) predictions. All other 
    terms do not correspond to a hard flip on the first measurement up to p**2 probability.

    Args:
    - processed_data (dict): A dictionary containing processed data for each qubit, as returned by
      `soft_postselect_calib_data`. Includes for each qubit: filtered measurement results
      ('mmr_0', 'mmr_1', 'mmr_0_scnd', 'mmr_1_scnd'), the fitted GMM
      object ('gmm'), and the scaler object ('scaler').

    Returns:
    - dict: A dictionary with qubit indices as keys. Each value is another dictionary with keys 'mmr_0' and
      'mmr_1', containing postselected data for initial measurements except those instances where both 
      the initial and final measurements are incorrect.

    This function uses GMM predictions to identify instances where both the initial and final measurements
    disagree with the expected outcomes. It excludes these instances, retaining all other measurements.
    """
    final_data = {}

    for qubit_idx, data in processed_data.items():
        gmm = data['gmm']
        scaler = data['scaler']

        # Predict labels for both initial and second measurements
        labels_0 = gmm.predict(scaler.transform(data['mmr_0'].real.reshape(-1, 1)))
        labels_1 = gmm.predict(scaler.transform(data['mmr_1'].real.reshape(-1, 1)))
        labels_0_scnd = gmm.predict(scaler.transform(data['mmr_0_scnd'].real.reshape(-1, 1)))
        labels_1_scnd = gmm.predict(scaler.transform(data['mmr_1_scnd'].real.reshape(-1, 1)))

        # Identify where either the initial or final measurement (or both) are correct
        # This excludes cases where both initial and final measurements are wrong
        correct_or_mixed_0 = ~((labels_0 == 1) & (labels_0_scnd == 1))
        correct_or_mixed_1 = ~((labels_1 == 0) & (labels_1_scnd == 0))

        # Apply postselection
        postselected_mmr_0 = data['mmr_0'][correct_or_mixed_0]
        postselected_mmr_1 = data['mmr_1'][correct_or_mixed_1]

        # Store postselected data
        final_data[qubit_idx] = {
            'mmr_0': postselected_mmr_0,
            'mmr_1': postselected_mmr_1
        }

    return final_data


def calculate_filtered_ratios(processed_data: dict):
    """
    Calculates the ratios of filtered measurement outcomes based on desired outcomes after GMM-based postselection
    filtering applied in `soft_postselect_calib_data`.

    Parameters:
    - processed_data (dict): A dictionary containing processed data for each qubit, as returned by
      `soft_postselect_calib_data`. It's expected to include for each qubit: filtered measurement results
      ('mmr_0', 'mmr_1', 'mmr_0_scnd', 'mmr_1_scnd'), the fitted GMM
      object ('gmm'), and the scaler object ('scaler').

    Returns:
    - dict: A dictionary with qubit indices as keys and another dictionary as values, which contains ratios
      of measurement outcomes that match the desired outcomes ('ratio_0_00', 'ratio_1_00', 'ratio_0_01', etc.),
      calculated from the filtered data.

    This function computes the ratios of the counts of specific outcomes (e.g., 00, 01, 10, 11) to the initial
    counts of '0' and '1' states for each qubit, using labels predicted by the GMM. It allows for evaluating
    the effect of filtering on the distribution of measurement outcomes.
    """
    ratio_results = {}


    for qubit_idx, data in processed_data.items():
        # Calculate the initial sizes
        initial_size_0 = data['mmr_0'].shape[0]
        initial_size_1 = data['mmr_1'].shape[0]

        # Retrieve the GMM and scaler
        gmm = data['gmm']
        scaler = data['scaler']

        # Predict labels for first and second measurements
        countMat_0 = gmm.predict(scaler.transform(data['mmr_0'].real.reshape(-1, 1)))
        countMat_0_scnd = gmm.predict(scaler.transform(data['mmr_0_scnd'].real.reshape(-1, 1)))
        countMat_1 = gmm.predict(scaler.transform(data['mmr_1'].real.reshape(-1, 1)))
        countMat_1_scnd = gmm.predict(scaler.transform(data['mmr_1_scnd'].real.reshape(-1, 1)))
    
        ratios = {}
        for desired_outcomes in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            # Define masks based on desired outcomes
            filter_mask_0 = (countMat_0 == desired_outcomes[0]) & (countMat_0_scnd == desired_outcomes[1])
            filter_mask_1 = (countMat_1 == desired_outcomes[0]) & (countMat_1_scnd == desired_outcomes[1])

            # Calculate the final sizes after applying the mask
            final_size_0 = data['mmr_0'][filter_mask_0].shape[0]
            final_size_1 = data['mmr_1'][filter_mask_1].shape[0]

            # Calculate the ratios
            ratio_0 = final_size_0 / initial_size_0 if initial_size_0 > 0 else 0
            ratio_1 = final_size_1 / initial_size_1 if initial_size_1 > 0 else 0

            # Store the results
            ratios[f"ratio_0_{desired_outcomes[0]}{desired_outcomes[1]}"] = ratio_0
            ratios[f"ratio_1_{desired_outcomes[0]}{desired_outcomes[1]}"] = ratio_1

        # Store in the results
        ratio_results[qubit_idx] = ratios

    return ratio_results


def get_msmt_err_probs(ratio_results):
    '''Calculate the measurement error probabilities from the ratio results.

    Args:
    - ratio_results (dict): A dictionary containing the ratio results for each qubit, as returned by
      `calculate_filtered_ratios`.
    
    Returns:
    - dict: A dictionary with qubit indices as keys. Each value is another dictionary with keys 'p_hard' and
      'p_soft', containing the estimated error probabilities for hard and soft flips, respectively.
    '''
    msmt_err_probs = {}
    for qubit_idx, ratios in ratio_results.items():
        msmt_err_probs[qubit_idx] = {
            'p_hard': (ratios['ratio_0_11'] + ratios['ratio_1_00']) / 2,
            'p_soft': (ratios['ratio_0_10'] + ratios['ratio_1_01']) / 2,
        }

    return msmt_err_probs



def filter_calib_data(processed_data: dict, desired_errs: str):
    filtered_results = {}

    for qubit_idx, data in processed_data.items():
        # Retrieve the GMM and scaler
        gmm = data['gmm']
        scaler = data['scaler']

        # Convert desired errors to desired outcomes (1 for agreement, 0 for disagreement)
        desired_outcomes = [int(err) for err in desired_errs]

        # Predict labels for first and second measurements
        countMat_0 = gmm.predict(scaler.transform(data['mmr_0'].real.reshape(-1, 1)))
        countMat_0_scnd = gmm.predict(scaler.transform(data['mmr_0_scnd'].real.reshape(-1, 1)))
        countMat_1 = gmm.predict(scaler.transform(data['mmr_1'].real.reshape(-1, 1)))
        countMat_1_scnd = gmm.predict(scaler.transform(data['mmr_1_scnd'].real.reshape(-1, 1)))

        # Define masks based on desired outcomes
        # For mmr_0: '0' means we expect a match with the label 0, '1' means a match with the label 1
        filter_mask_0 = ((countMat_0 == 0) == desired_outcomes[0]) & ((countMat_0_scnd == 0) == desired_outcomes[1])
        filter_mask_1 = ((countMat_1 == 1) == desired_outcomes[0]) & ((countMat_1_scnd == 1) == desired_outcomes[1])

        # Apply the filters
        mmr_0 = data['mmr_0'][filter_mask_0]
        mmr_0_scnd = data['mmr_0_scnd'][filter_mask_0]
        mmr_1 = data['mmr_1'][filter_mask_1]
        mmr_1_scnd = data['mmr_1_scnd'][filter_mask_1]

        # Store in the results
        filtered_results[qubit_idx] = {
            'mmr_0': mmr_0,
            'mmr_0_scnd': mmr_0_scnd,
            'mmr_1': mmr_1,
            'mmr_1_scnd': mmr_1_scnd
        }

    return filtered_results


def calculate_filtered_ratio(processed_data: dict, desired_errs: str):
    ratio_results = {}

    for qubit_idx, data in processed_data.items():
        # Calculate the initial sizes
        initial_size_0 = data['mmr_0'].shape[0]
        initial_size_1 = data['mmr_1'].shape[0]

        # Retrieve the GMM and scaler
        gmm = data['gmm']
        scaler = data['scaler']

        # Convert desired errors to desired outcomes (1 for agreement, 0 for disagreement)
        desired_outcomes = [int(err) for err in desired_errs]

        # Predict labels for first and second measurements
        countMat_0 = gmm.predict(scaler.transform(data['mmr_0'].real.reshape(-1, 1)))
        countMat_0_scnd = gmm.predict(scaler.transform(data['mmr_0_scnd'].real.reshape(-1, 1)))
        countMat_1 = gmm.predict(scaler.transform(data['mmr_1'].real.reshape(-1, 1)))
        countMat_1_scnd = gmm.predict(scaler.transform(data['mmr_1_scnd'].real.reshape(-1, 1)))

        # Define masks based on desired outcomes
        filter_mask_0 = ((countMat_0 == 0) == desired_outcomes[0]) & ((countMat_0_scnd == 0) == desired_outcomes[1])
        filter_mask_1 = ((countMat_1 == 1) == desired_outcomes[0]) & ((countMat_1_scnd == 1) == desired_outcomes[1])

        # Calculate the final sizes after applying the mask
        final_size_0 = data['mmr_0'][filter_mask_0].shape[0]
        final_size_1 = data['mmr_1'][filter_mask_1].shape[0]

        # Calculate the ratios
        ratio_0 = final_size_0 / initial_size_0 if initial_size_0 > 0 else 0
        ratio_1 = final_size_1 / initial_size_1 if initial_size_1 > 0 else 0

        # Store in the results
        ratio_results[qubit_idx] = {
            'ratio_0': ratio_0,
            'ratio_1': ratio_1
        }

    return ratio_results



