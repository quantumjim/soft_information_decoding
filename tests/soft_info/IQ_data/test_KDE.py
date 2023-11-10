from unittest import mock

import numpy as np
from numpy.random import default_rng
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from src.soft_info.IQ_data.KDE import fit_KDE, get_KDEs


def generate_mock_IQ_data(seed=42, size=(1000, 2), mean=0, std=1):
    rng = default_rng(seed)  # Create a random generator with a fixed seed
    real_part = rng.normal(mean, std, size)  # Gaussian distributed real parts
    # Gaussian distributed imaginary parts
    imag_part = rng.normal(mean, std, size)
    return real_part + 1j * imag_part


@mock.patch('src.soft_info.IQ_data.plotter.plot_IQ_data')
@mock.patch('src.soft_info.IQ_data.KDE.plot_KDE')
def test_fit_KDE(mock_plot_IQ_data, mock_plot_KDE):
    # Create mock IQ data
    mock_IQ_data = generate_mock_IQ_data()

    # Call the function
    kde, scaler = fit_KDE(mock_IQ_data, bandwidth=0.01,
                          plot=False, qubit_index='Q1', num_samples=10)

    # Assertions
    assert isinstance(
        kde, KernelDensity), "Returned object is not a KDE instance"
    assert isinstance(
        scaler, StandardScaler), "Returned object is not a StandardScaler instance"

    # Flatten and split the IQ data
    data = mock_IQ_data.flatten()
    combined_data = np.column_stack((data.real, data.imag))
    normalized_data = scaler.transform(combined_data)

    # Check the score (log-likelihood) of the original data
    normalized_data = scaler.transform(combined_data)
    neg_llh = kde.score_samples(normalized_data)
    assert np.mean(neg_llh) > -0.20, "KDE does not fit the data well"

    # Check if the KDE object is fitted
    assert hasattr(kde, 'tree_'), "KDE object is not fitted"

    # Check if the plotting functions were not called
    mock_plot_IQ_data.assert_not_called()
    mock_plot_KDE.assert_not_called()



# @mock.patch('src.soft_info.IQ_data.KDE.fit_KDE')
# @mock.patch('src.Scratch.calibration_data.load_calibration_memory')
# def test_get_KDEs(mock_load_calibration_memory, mock_fit_KDE):
#     # Create mock calibration memory data
#     mock_memories = {
#         qubit: {
#             "mmr_0": np.random.rand(10) + 1j * np.random.rand(10),
#             "mmr_1": np.random.rand(10) + 1j * np.random.rand(10)
#         }
#         for qubit in [0, 1]
#     }
#     mock_load_calibration_memory.return_value = mock_memories

#     # Mock fit_KDE to return a KDE and scaler for each call
#     mock_fit_KDE.return_value = (mock.Mock(), mock.Mock())

#     # Call the function
#     provider_mock = mock.Mock()
#     device = 'device_name'
#     qubits = [0, 1]
#     all_kdes, all_scalers = get_KDEs(provider_mock, device, qubits)

#     # Debug print
#     print("Mock Memories:", mock_memories)
#     print("All KDEs:", all_kdes)
#     print("All Scalers:", all_scalers)

#     # Assertions
#     assert len(all_kdes) == len(
#         qubits), "Number of KDEs does not match number of qubits"
#     assert len(all_scalers) == len(
#         qubits), "Number of scalers does not match number of qubits"
#     for qubit in qubits:
#         assert len(all_kdes[qubit]) == 2, "Each qubit should have two KDEs"
#         assert isinstance(all_scalers[qubit],
#                           mock.Mock), "Scaler is not a mock object"

#     # Verify if the mocked functions are called correctly
#     mock_load_calibration_memory.assert_called_with(
#         provider_mock, device, qubits)
#     mock_fit_KDE.assert_called()
