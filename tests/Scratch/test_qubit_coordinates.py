import pytest
from unittest import mock
from src.Scratch.qubit_coordinates import get_qubit_coordinates
import json

@mock.patch('src.Scratch.core.find_and_create_scratch')
@mock.patch('builtins.open', new_callable=mock.mock_open, read_data=json.dumps({"2": [(1, 2), (3, 4)]}))
def test_get_qubit_coordinates_without_rotation(mock_open, mock_find_and_create_scratch):
    """
    Test get_qubit_coordinates function without rotation.
    """
    mock_find_and_create_scratch.return_value = '/fake/path'
    
    coordinates = get_qubit_coordinates(2, rotated=False)
    assert coordinates == [(1, 2), (3, 4)], "Coordinates should match the JSON data"

@mock.patch('src.Scratch.core.find_and_create_scratch')
@mock.patch('builtins.open', new_callable=mock.mock_open, read_data=json.dumps({"2": [[1, 2], [3, 4]]}))
def test_get_qubit_coordinates_without_rotation(mock_open, mock_find_and_create_scratch):
    """
    Test get_qubit_coordinates function without rotation.
    """
    mock_find_and_create_scratch.return_value = '/fake/path'
    
    coordinates = get_qubit_coordinates(2, rotated=False)
    assert coordinates == [[1, 2], [3, 4]], "Coordinates should match the JSON data"

