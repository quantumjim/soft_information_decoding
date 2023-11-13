import pytest
from unittest import mock
from src.Scratch.calibration_data import load_calibration_memory
import pandas as pd

@pytest.fixture
def mock_provider():
    """Creates a mock provider with a mock job and result."""
    mock_result = mock.Mock()
    mock_result.get_memory.return_value = ...
    # Define the memory data structure that get_memory should return

    mock_job = mock.Mock()
    mock_job.result.return_value = mock_result

    provider = mock.Mock()
    provider.retrieve_job.return_value = mock_job

    return provider

@mock.patch('src.Scratch.metadata.metadata_loader')
@pytest.mark.filterwarnings(r"ignore:Loaded 0 memories with keys \[\]:UserWarning")
def test_no_matching_metadata_entries(mock_metadata_loader, mock_provider):
    mock_metadata_loader.return_value = pd.DataFrame()
    # Create a dataframe that has no matching entries for the filters

    qubits = [0, 1]
    memories = load_calibration_memory(mock_provider, 'device', qubits)
    assert all(len(mem) == 0 for mem in memories.values()), "No memories should be loaded"

