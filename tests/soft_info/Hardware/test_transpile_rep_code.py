from unittest import mock
import pytest

from src.soft_info.Hardware.transpile_rep_code import get_repcode_layout, get_repcode_IQ_map

@mock.patch('src.soft_info.Hardware.coupling_map.find_longest_path_in_hex')
@mock.patch('src.soft_info.Hardware.coupling_map.find_longest_path_general')
def test_get_repcode_layout(mock_find_general, mock_find_hex):
    # Create a mock backend
    backend_mock = mock.Mock()

    # Create a mock configuration
    mock_configuration = mock.Mock()
    mock_configuration.coupling_map = [[0, 1], [1, 2], [2, 3], [3, 4]]  # Example coupling map
    mock_configuration.n_qubits = 5

    # Set the backend mock to return this configuration
    backend_mock.configuration.return_value = mock_configuration

    # Mock the longest path functions
    mock_path = [0, 1, 2, 3, 4]
    mock_find_hex.return_value = (mock_path, len(mock_path), 0)
    mock_find_general.return_value = (mock_path, len(mock_path), 0)

    # Test with hex topology
    layout_hex = get_repcode_layout(3, backend_mock, _is_hex=True)
    assert layout_hex == [3, 1, 4, 2, 0], "Layout for hex topology is incorrect"

    # Test with general topology
    layout_general = get_repcode_layout(3, backend_mock, _is_hex=False)
    assert layout_general == [1, 3, 0, 2, 4], "Layout for general topology is incorrect"

    # Test with distance exceeding path length
    with pytest.raises(ValueError):
        get_repcode_layout(4, backend_mock, _is_hex=True)


def test_get_repcode_IQ_map():
    layout = [1, 3, 5, 0, 2, 4, 6]
    synd_rounds = 2
    n_link_qubits = len(layout) // 2

    iq_map = get_repcode_IQ_map(layout, synd_rounds)

    expected_iq_map = {}
    # Link qubits for each round
    for t in range(synd_rounds):
        for idx in range(n_link_qubits):
            expected_iq_map[t * n_link_qubits + idx] = layout[idx]

    # Code qubits
    for idx in range(n_link_qubits+1):
        expected_iq_map[synd_rounds * n_link_qubits + idx] = layout[n_link_qubits + idx]

    assert iq_map == expected_iq_map, "IQ map is incorrect"
