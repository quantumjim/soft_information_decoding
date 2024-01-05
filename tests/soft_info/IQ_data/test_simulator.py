from unittest.mock import Mock, patch, MagicMock
import unittest

import pytest
import numpy as np
from qiskit_qec.circuits import RepetitionCodeCircuit
from qiskit_qec.noise import PauliNoiseModel

from soft_info import RepCodeIQSimulator  
from Scratch.calibration.calibration_data import find_closest_calib_jobs

    
class TestRepCodeIQSimulator(unittest.TestCase):    

    # Patch where the function is imported not where it is defined
    @patch('soft_info.IQ_data.simulator.get_repcode_IQ_map')
    @patch('soft_info.IQ_data.simulator.get_repcode_layout')  
    @patch('soft_info.IQ_data.simulator.get_KDEs')  
    @patch('soft_info.IQ_data.simulator.create_or_load_kde_grid')
    def test_constructor(self, mock_create_or_load_kde_grid, mock_get_KDEs, mock_get_repcode_layout, mock_get_repcode_IQ_map):
        # Set up the mock return values
        mock_provider = MagicMock()
        mock_get_repcode_layout.return_value = [0, 1, 2, 3, 4]
        mock_get_repcode_IQ_map.return_value = MagicMock()      
        mock_get_KDEs.return_value = (MagicMock(), MagicMock())
        mock_create_or_load_kde_grid.return_value = (MagicMock(), MagicMock())

        # Parameters for the RepCodeIQSimulator
        distance = 2
        rounds = 5
        device = "mock_device"
        _is_hex = False
        other_date = None

        # Create the RepCodeIQSimulator instance
        simulator = RepCodeIQSimulator(mock_provider, distance, rounds, device, _is_hex, other_date=other_date)

        # Assertions
        self.assertEqual(simulator.distance, distance)
        self.assertEqual(simulator.rounds, rounds)
        self.assertEqual(simulator.device, device)
        self.assertEqual(simulator.provider, mock_provider)
        self.assertTrue(isinstance(simulator.backend, MagicMock))
        self.assertEqual(simulator.layout, mock_get_repcode_layout.return_value)
        self.assertTrue(isinstance(simulator.qubit_mapping, MagicMock))
        self.assertTrue(isinstance(simulator.kde_dict, MagicMock))
        self.assertTrue(isinstance(simulator.scaler_dict, MagicMock))
        self.assertTrue(isinstance(simulator.grid_dict, MagicMock))
        self.assertTrue(isinstance(simulator.processed_scaler_dict, MagicMock))

        # Validate that the mocks were called correctly
        mock_get_repcode_IQ_map.assert_called_with(mock_get_repcode_layout.return_value, rounds)
        mock_get_KDEs.assert_called_with(mock_provider, tobecalib_backend=device, other_date=other_date)
        mock_create_or_load_kde_grid.assert_called_with(mock_provider, tobecalib_backend=device,
                                                        num_grid_points=300, num_std_dev=2, other_date=other_date)
        

    @patch.object(RepCodeIQSimulator, '__init__', lambda x, *args, **kwargs: None)
    def test_counts_to_IQ_two_qubit_system(self):
        # Mock KDEs and scalers
        mock_kde0 = MagicMock()
        mock_kde1 = MagicMock()
        mock_scaler = MagicMock()

        # Define predefined samples for KDEs
        mock_kde0.sample.return_value = np.array([[0.1, 0.2], [0.6, 0.8], [0.3, 0.4], [0.4, 0.6]])
        mock_kde1.sample.return_value = np.array([[0.5, 0.6], [0.7, 0.8], [0.1, 0.2], [0.1, 0.2]])
        mock_scaler.inverse_transform.side_effect = lambda x: x  # Identity function for simplicity

        # Set up the RepCodeIQSimulator instance
        simulator = RepCodeIQSimulator()
        simulator.kde_dict = {0: [mock_kde0, mock_kde1], 1: [mock_kde0, mock_kde1], 2: [mock_kde0, mock_kde1]}
        simulator.scaler_dict = {0: mock_scaler, 1: mock_scaler, 2: mock_scaler}
        simulator.qubit_mapping = [1, 0, 2]  
        simulator.rounds = 1

        # Count data for '01 1', '11 0', and '00 0'
        counts = {"01 1": 3, "11 0": 1, "00 0": 1}

        # Call the method
        iq_memory = simulator.counts_to_IQ(counts)

        # Expected output calculation
        expected_iq_memory = np.array([
            [0.5+0.6j, 0.5+0.6j, 0.1+0.2j], # 1 10
            [0.7+0.8j, 0.7+0.8j, 0.6+0.8j], # 1 10
            [0.1+0.2j, 0.1+0.2j, 0.3+0.4j], # 1 10
            [0.1+0.2j, 0.1+0.2j, 0.5+0.6j], # 0 11
            [0.6+0.8j, 0.1+0.2j, 0.4+0.6j], # 0 00
        ])

        # Assertions
        np.testing.assert_array_almost_equal(iq_memory, expected_iq_memory, decimal=5)
        mock_kde0.sample.assert_called()
        mock_kde1.sample.assert_called()
        mock_scaler.inverse_transform.assert_called()


    @patch.object(RepCodeIQSimulator, '__init__', lambda x, *args, **kwargs: None)
    @patch('soft_info.IQ_data.simulator.RepCodeIQSimulator.get_counts')
    def test_counts_to_IQ_extreme(self, mock_get_counts):

        # Mock set up
        mock_get_counts.return_value = {"00 0": 1}
        mock_IQ_dict = {
            0: {"iq_point_safe": (0.09+0.09j), "iq_point_ambig": (0+0j)},
            1: {"iq_point_safe": (0.1+0.1j), "iq_point_ambig": (1+1j)},
            2: {"iq_point_safe": (0.2+0.2j), "iq_point_ambig": (2+2j)},
            }

        # Set up the RepCodeIQSimulator instance
        simulator = RepCodeIQSimulator()
        simulator.qubit_mapping = [0, 2, 1]  
        simulator.rounds = 1

        iq_memory_safe = simulator.counts_to_IQ_extreme(shots=1, IQ_dict=mock_IQ_dict, p_ambig=0)
        iq_memory_ambig = simulator.counts_to_IQ_extreme(shots=1, IQ_dict=mock_IQ_dict, p_ambig=1)

        # Assertions
        np.testing.assert_array_almost_equal(iq_memory_safe, np.array([[0.09+0.09j, 0.2+0.2j, 0.1+0.1j]]), decimal=5)
        np.testing.assert_array_almost_equal(iq_memory_ambig, np.array([[0+0j, 0.2+0.2j, 0.1+0.1j]]), decimal=5)





if __name__ == '__main__':
    unittest.main()
