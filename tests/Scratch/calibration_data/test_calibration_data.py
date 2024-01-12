import pytest
from unittest import mock
from unittest.mock import patch
from Scratch.calibration.calibration_data import find_closest_calib_jobs
import pandas as pd
import numpy as np

@pytest.fixture
def mock_provider():
    """Creates a mock provider with a mock job and result."""
    # Define a mock memory structure - adjust this as per your actual data
    mock_memory = np.array([[1, 0], [0, 1]])

    mock_result = mock.Mock()
    mock_result.get_memory.return_value = mock_memory

    mock_job = mock.Mock()
    mock_job.result.return_value = mock_result

    provider = mock.Mock()
    provider.retrieve_job.return_value = mock_job

    return provider

def test_not_implemented_error():
    with pytest.raises(NotImplementedError):
        find_closest_calib_jobs()

@patch('Scratch.calibration.calibration_data.get_calib_jobs')  
def test_loading_newest_jobs(mock_get_calib_jobs):
    backend_name = 'test_backend'
    mock_get_calib_jobs.return_value = ({'0': 'job_id_0', '1': 'job_id_1'}, 
                                        {'0': pd.to_datetime('2023-01-01T00:00:00', utc=True),
                                         '1': pd.to_datetime('2023-01-01T01:00:00', utc=True)},
                                        {'0': pd.to_datetime('2023-01-01T00:00:00', utc=True),
                                         '1': pd.to_datetime('2023-01-01T01:00:00', utc=True)})

    expected_job_ids = {'0': 'job_id_0', '1': 'job_id_1'}
    job_ids, backend, creation_dates = find_closest_calib_jobs(tobecalib_backend=backend_name)
    
    assert job_ids == expected_job_ids
    assert backend == backend_name
