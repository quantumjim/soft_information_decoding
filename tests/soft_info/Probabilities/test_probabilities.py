from unittest import mock
import warnings

import numpy as np
from qiskit.result import Counts
import pytest
from sklearn.preprocessing import StandardScaler

from src.soft_info.Probabilities.probabilities import estimate_outcome, get_counts, llh_ratio


def test_estimate_outcome():
    # Mock the KDE models and scaler
    kde_0_mock = mock.Mock()
    kde_1_mock = mock.Mock()
    scaler_mock = StandardScaler()
    scaler_mock.fit([[0, 0]])  # Fit with dummy data

    # Case 1: IQ_point closer to kde_0
    real_part = 0.1 
    imag_part = 0.1
    IQ_point = real_part + 1j * imag_part

    # Correctly reshape IQ_point for the scaler
    kde_0_mock.score_samples.return_value = np.array([0.6])
    kde_1_mock.score_samples.return_value = np.array([0.4])
    assert estimate_outcome(IQ_point, kde_0=kde_0_mock, kde_1=kde_1_mock, scaler=scaler_mock) == 0


    # Case 2: IQ_point closer to kde_1
    kde_0_mock.score_samples.return_value = np.array([0.3])
    kde_1_mock.score_samples.return_value = np.array([0.7])
    assert estimate_outcome(IQ_point, kde_0=kde_0_mock, kde_1=kde_1_mock, scaler=scaler_mock) == 1

    # Case 3: Fallback (no KDE models or scaler provided)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        IQ_point = np.array([-0.1 + 0.1j])  # Real part is negative
        assert estimate_outcome(IQ_point) == 0
        assert len(w) == 1  # Check that a warning was raised

        IQ_point = np.array([0.1 + 0.1j])  # Real part is positive
        assert estimate_outcome(IQ_point) == 1
        assert len(w) == 2  # Check that another warning was raised


# Mocking necessary functions and classes
@pytest.fixture
def mock_kde_dict():
    # Mock KDE models for qubit states
    return {
        0: (mock.Mock(), mock.Mock()),  # KDEs for qubit 0
        1: (mock.Mock(), mock.Mock())   # KDEs for qubit 1
    }

@pytest.fixture
def mock_scaler_dict():
    # Mock scalers for qubits
    return {
        0: mock.Mock(),  # Scaler for qubit 0
        1: mock.Mock()   # Scaler for qubit 1
    }

@pytest.fixture
def mock_layout():
    # Example layout
    return [1, 3, 1, 2, 4]  # Link qubits followed by code qubits

def test_get_counts(mock_kde_dict, mock_scaler_dict, mock_layout):
    n_links = len(mock_layout) // 2

    # Example syndrome rounds
    synd_rounds = 2

    # Mock IQ data for a few shots
    mock_IQ_data = [
        [np.random.rand() + 1j * np.random.rand() for _ in range(n_links*synd_rounds + n_links+1)],
        [np.random.rand() + 1j * np.random.rand() for _ in range(n_links*synd_rounds + n_links+1)]
    ]
    
    # Mock the estimate_outcome function
    with mock.patch('src.soft_info.Probabilities.probabilities.estimate_outcome', return_value=0):
        # Call the get_counts function
        counts = get_counts(mock_IQ_data, kde_dict=mock_kde_dict, scaler_dict=mock_scaler_dict, 
                            layout=mock_layout, synd_rounds=synd_rounds, verbose=False)
        print("mock counts:", counts)
    
    # Assertions
    assert isinstance(counts, Counts), "The function should return an instance of qiskit.result.Counts"
    for key in counts.keys():
        assert all(char in ['0', '1', ' '] for char in key), "Each character in count keys should be 0, 1, or space"


def test_get_counts_no_kde(mock_layout):
    n_links = len(mock_layout) // 2

    # Example syndrome rounds
    synd_rounds = 2
    
    mock_IQ_data = [
        [1+0j for _ in range(n_links*synd_rounds + n_links+1)],
        [-1+0j for _ in range(n_links*synd_rounds + n_links+1)]
    ]
    
    # Expecting a UserWarning due to missing KDEs or scaler
    with pytest.warns(UserWarning, match="Not enough kernels or no scaler provided"):
        counts = get_counts(mock_IQ_data, layout=mock_layout, synd_rounds=synd_rounds, verbose=False)

    assert isinstance(counts, Counts), "The function should return an instance of qiskit.result.Counts"
    assert counts == {"111 11 11": 1, "000 00 00": 1}, "The function should return the correct counts"


# def test_llh_ratio():
#     # Mock the IQ point, KDE models, and scaler
#     IQ_point = np.array([0.5 + 0.5j])
#     kde_0_mock = KernelDensity()
#     kde_1_mock = KernelDensity()
#     scaler_mock = StandardScaler()

#     # Fit mock data for the scaler and KDE models
#     mock_data = np.array([[0+ 0j], [1 + 1j]])
#     scaler_mock.fit(mock_data)
#     kde_0_mock.fit(scaler_mock.transform(mock_data))
#     kde_1_mock.fit(scaler_mock.transform(mock_data))

#     # Calculate log likelihood ratio
#     llh_ratio_result = llh_ratio(IQ_point, kde_0=kde_0_mock, kde_1=kde_1_mock, scaler=scaler_mock)

#     # Check if the result is valid
#     assert llh_ratio_result is not None, "Log likelihood ratio should be computed"
#     assert isinstance(llh_ratio_result, float), "Log likelihood ratio should be a float"

#     # Check for expected behavior when KDEs or scaler are missing
#     with pytest.warns(UserWarning, match="Not enough kernels or no scaler provided"):
#         assert llh_ratio(None, kde_0=None, kde_1=None, scaler=None) is None, "Should return None when kernels or scaler are missing"
