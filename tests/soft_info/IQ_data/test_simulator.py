from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np
from qiskit_qec.circuits import RepetitionCodeCircuit
from qiskit_qec.noise import PauliNoiseModel

from soft_info import RepCodeIQSimulator  
from Scratch.calibration.calibration_data import find_closest_calib_jobs


# class TestRepCodeIQSimulator:

#     @patch('result_saver.SaverProvider.get_backend')  # Adjusted import path
#     @patch('soft_info.get_repcode_layout')  # Adjusted import path
#     @patch('soft_info.get_repcode_IQ_map')  # Adjusted import path
#     @patch('soft_info.get_KDEs')  # Adjusted import path
#     @patch('Scratch.calibration.calibration_data.load_calibration_memory')
#     @patch('Scratch.calibration.calibration_data.get_calib_jobs')
#     @patch('Scratch.calibration.calibration_data.find_closest_calib_jobs')
#     @patch('Scratch.calibration.calib_metadata.create_or_load_kde_grid')
#     def test_init(self, mock_create_or_load_kde_grid, mock_find_closest_calib_jobs, mock_get_calib_jobs, 
#               mock_load_calibration_memory, mock_get_KDEs, mock_get_repcode_IQ_map, 
#               mock_get_repcode_layout, mock_get_backend):
#         # Mock values
#         mock_create_or_load_kde_grid.return_value = (MagicMock(), MagicMock())
#         mock_provider = Mock()
#         mock_backend = Mock()
#         mock_configuration = MagicMock()
#         mock_configuration.coupling_map = [(0, 1), (1, 2), (2, 3)]  # Example coupling map
#         mock_backend.configuration.return_value = mock_configuration
#         mock_get_backend.return_value = mock_backend
#         mock_provider.get_backend = mock_get_backend
#         mock_load_calibration_memory.return_value = {"dummy_key": "dummy_value"}
#         mock_get_calib_jobs.return_value = (
#             {'0': 'job_id_0', '1': 'job_id_1'},  # job_ids
#             {'0': 'execution_date_0', '1': 'execution_date_1'},  # execution_dates
#             {'0': 'creation_date_0', '1': 'creation_date_1'}  # creation_dates
#         )

#         # Mock find_closest_calib_jobs and its dependencies
#         mock_job = Mock()
#         mock_result = Mock()
#         mock_memory = np.array([[1, 0], [0, 1]])  # Example mock memory
#         mock_result.get_memory.return_value = mock_memory
#         mock_job.result.return_value = mock_result
#         mock_provider.retrieve_job.return_value = mock_job
#         mock_find_closest_calib_jobs.return_value = ({"0": "job_id_0", "1": "job_id_1"}, {"0": "execution_date_0", "1": "execution_date_1"}, {"0": "creation_date_0", "1": "creation_date_1"})


#         mock_get_repcode_layout.return_value = Mock()
#         mock_get_repcode_IQ_map.return_value = Mock()
#         mock_get_KDEs.return_value = (Mock(), Mock())

#         # Parameters for initialization
#         distance = 2
#         rounds = 5
#         device = "mock_device"
#         _is_hex = True
#         other_date = None

#         # Create an instance of RepCodeIQSimulator
#         simulator = RepCodeIQSimulator(mock_provider, distance, rounds, device, _is_hex, other_date)
#         simulator.grid_dict = MagicMock()
#         simulator.processed_scaler_dict = MagicMock()

#         # Assertions to validate the initialization
#         assert simulator.distance == distance
#         assert simulator.rounds == rounds
#         assert simulator.device == device
#         assert simulator.provider is mock_provider
#         assert simulator.backend is mock_backend
#         # assert simulator.layout is mock_get_repcode_layout.return_value
#         # assert simulator.qubit_mapping is mock_get_repcode_IQ_map.return_value
#         # assert simulator.kde_dict is mock_get_KDEs.return_value[0]
#         # assert simulator.scaler_dict is mock_get_KDEs.return_value[1]
#         assert isinstance(simulator.code, RepetitionCodeCircuit)

#         # Validate that the mocks were called correctly
#         mock_get_backend.assert_called_with(device)
#         # mock_get_repcode_layout.assert_called_with(distance, mock_backend, _is_hex=_is_hex)
#         # mock_get_repcode_IQ_map.assert_called_with(mock_get_repcode_layout.return_value, rounds)
#         # mock_get_KDEs.assert_called_with(mock_provider, tobecalib_backend=device, other_date=other_date)
#         # mock_find_closest_calib_jobs.assert_called()
#         # mock_get_calib_jobs.assert_called()
#         # mock_load_calibration_memory.assert_called()

#         #TODO: Fix all the other asserts

        
