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

if __name__ == '__main__':
    unittest.main()
