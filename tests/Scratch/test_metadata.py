from unittest import mock

import pytest
import pandas as pd
import json

from src.Scratch.metadata import metadata_loader, metadata_helper, update_metadata


def test_metadata_loader_default():
    """
    Test metadata_loader with default parameters.
    """
    # Execute
    result = metadata_loader()

    # Assert
    assert isinstance(
        result, pd.DataFrame), "Result should be a pandas DataFrame"
    # Add more assertions here to validate the contents of the DataFrame


def test_metadata_loader_extract():
    """
    Test metadata_loader with _extract set to True.
    """

    # Execute
    result = metadata_loader(_extract=True)

    # Assert
    assert isinstance(
        result, pd.DataFrame), "Result should be a pandas DataFrame"
    # Add more assertions here to validate the extraction logic


def test_metadata_loader_file_not_found():
    with mock.patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError) as excinfo:
            metadata_loader()
        assert "job_metadata.json file not found." in str(excinfo.value)


def test_metadata_loader_file_found():
    # Mock os.path.exists to return True and pandas.read_json to return a mock DataFrame
    mock_data = {
        'column1': [1, 2, 3],
        # Adding 'creation_date'
        'creation_date': ['2021-01-01', '2021-01-02', '2021-01-03']
    }
    mock_df = pd.DataFrame(mock_data)

    with mock.patch('os.path.exists', return_value=True):
        with mock.patch('pandas.read_json', return_value=mock_df):
            result = metadata_loader()
            assert isinstance(
                result, pd.DataFrame), "Result should be a pandas DataFrame"
            # Further assertions as needed


def test_metadata_helper_no_args():
    """
    Test metadata_helper with no arguments.
    """
    result = metadata_helper()
    assert isinstance(result, dict), "Result should be a dictionary"
    assert result == {}, "Result should be an empty dictionary"


def test_metadata_helper_positional_args():
    """
    Test metadata_helper with positional arguments.
    """
    result = metadata_helper('arg1', 'arg2')
    assert result.get('additional_args') == (
        'arg1', 'arg2'), "Positional arguments should be included correctly"


def test_metadata_helper_keyword_args():
    """
    Test metadata_helper with keyword arguments.
    """
    result = metadata_helper(n_shots=1000, meas_level=2)
    assert result == {
        'n_shots': 1000, 'meas_level': 2}, "Keyword arguments should be included correctly"


def test_metadata_helper_both_args():
    """
    Test metadata_helper with both positional and keyword arguments.
    """
    result = metadata_helper('arg1', n_shots=1000, meas_level=2)
    assert result.get('additional_args') == (
        'arg1',), "Positional arguments should be included correctly"
    assert result.get('n_shots') == 1000 and result.get(
        'meas_level') == 2, "Keyword arguments should be included correctly"


@mock.patch('src.Scratch.metadata.find_and_create_scratch')
@mock.patch('src.Scratch.metadata.get_notebook_name')
@mock.patch('src.Scratch.metadata.open', new_callable=mock.mock_open)
@mock.patch('src.Scratch.metadata.os.path.exists')
@mock.patch('src.Scratch.metadata.json.dump')
@mock.patch('src.Scratch.metadata.json.load')
def test_update_metadata_new_entry(mock_json_load, mock_json_dump, mock_exists, mock_open, mock_get_notebook_name, mock_find_and_create_scratch):
    """
    Test update_metadata function for creating a new metadata entry.
    """
    # Setup mocks
    mock_find_and_create_scratch.return_value = "/fake/path"
    mock_get_notebook_name.return_value = "test_notebook"
    mock_exists.return_value = False
    job_mock = mock.Mock()
    job_mock.job_id.return_value = '123'
    job_mock.creation_date.return_value = '2021-01-01'
    job_mock.backend_options.return_value = {}
    job_mock.name.return_value = 'test_job'
    job_mock.metadata = {}
    job_mock.tags.return_value = ['tag1', 'tag2']

    # Call the function
    update_metadata(job_mock, 'fake_backend', {'key': 'value'})

    # Check if a new entry is created
    mock_open.assert_called_with('/fake/path/job_metadata.json', 'w')
    mock_json_dump.assert_called_once()
    args, kwargs = mock_json_dump.call_args
    assert len(args[0]) == 1  # One entry in the list
    assert args[0][0]['job_id'] == '123'
    assert args[0][0]['additional_metadata'] == {'key': 'value'}


@mock.patch('src.Scratch.metadata.find_and_create_scratch')
@mock.patch('src.Scratch.metadata.get_notebook_name')
@mock.patch('src.Scratch.metadata.open', new_callable=mock.mock_open, read_data=json.dumps([{'job_id': '123', 'additional_metadata': {}}]))
@mock.patch('src.Scratch.metadata.os.path.exists')
@mock.patch('src.Scratch.metadata.json.dump')
@mock.patch('src.Scratch.metadata.json.load')
def test_update_metadata_existing_entry(mock_json_load, mock_json_dump, mock_exists, mock_open, mock_get_notebook_name, mock_find_and_create_scratch):
    """
    Test update_metadata function for updating an existing metadata entry.
    """
    # Setup mocks
    mock_exists.return_value = True
    mock_json_load.return_value = [
        {'job_id': '123', 'additional_metadata': {}}]
    job_mock = mock.Mock()
    job_mock.job_id.return_value = '123'

    # Call the function
    update_metadata(job_mock, 'fake_backend', {'new_key': 'new_value'})

    # Check if the existing entry is updated
    mock_json_dump.assert_called_once()
    args, kwargs = mock_json_dump.call_args
    assert args[0][0]['job_id'] == '123'
    assert args[0][0]['additional_metadata'] == {'new_key': 'new_value'}
