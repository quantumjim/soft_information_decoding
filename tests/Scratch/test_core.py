from unittest import mock
import os
from src.Scratch.metadata import find_and_create_scratch, get_notebook_name

# Global variable for the current mocked path
current_path = '/path/to/Soft-Info'

@mock.patch('src.Scratch.metadata.os.getcwd')
@mock.patch('src.Scratch.metadata.os.chdir')
@mock.patch('src.Scratch.metadata.os.path.exists')
@mock.patch('src.Scratch.metadata.os.mkdir')
def test_find_and_create_scratch(mock_mkdir, mock_exists, mock_chdir, mock_getcwd):
    """
    Test find_and_create_scratch function.
    """

    def mock_getcwd_side_effect():
        global current_path
        return current_path

    mock_getcwd.side_effect = mock_getcwd_side_effect

    def mock_chdir_side_effect(new_directory):
        global current_path
        if new_directory == '..':
            current_path = os.path.dirname(current_path)
        else:
            current_path = os.path.join(current_path, new_directory)

    mock_chdir.side_effect = mock_chdir_side_effect

    def side_effect(path):
        global current_path
        full_path = os.path.join(current_path, path)
        required_paths = [
            '/path/to/Soft-Info/.git',
            '/path/to/Soft-Info/README.md',
            '/path/to/Soft-Info/src',
            '/path/to/Soft-Info/libs'
        ]
        return full_path in required_paths

    mock_exists.side_effect = side_effect

    # Now run your test logic
    result = find_and_create_scratch()
    assert result == '/path/to/Soft-Info/.Scratch'
    mock_mkdir.assert_called_once_with('.Scratch')
    

@mock.patch('inspect.currentframe')
def test_get_notebook_name(mock_currentframe):
    """
    Test get_notebook_name function.
    """
    mock_frame = mock.Mock()
    mock_frame.f_globals = {'__vsc_ipynb_file__': '/path/to/notebook.ipynb'}
    mock_frame.f_back = None  # Ensure the frame chain ends
    mock_currentframe.return_value = mock_frame

    result = get_notebook_name()
    assert result == 'notebook'
