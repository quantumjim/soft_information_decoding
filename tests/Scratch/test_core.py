from unittest import mock
from src.Scratch.metadata import find_and_create_scratch, get_notebook_name

@mock.patch('src.Scratch.metadata.os.getcwd')
@mock.patch('src.Scratch.metadata.os.chdir')
@mock.patch('src.Scratch.metadata.os.path.exists')
@mock.patch('src.Scratch.metadata.os.mkdir')
def test_find_and_create_scratch(mock_mkdir, mock_exists, mock_chdir, mock_getcwd):
    """
    Test find_and_create_scratch function.
    """
    mock_getcwd.return_value = '/path/to/Soft-Info'
    mock_exists.return_value = False

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
