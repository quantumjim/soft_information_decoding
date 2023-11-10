import networkx as nx
from src.soft_info.Hardware.coupling_map import add_qubit_positions, create_coupling_graph_with_positions, find_longest_path_general, find_longest_path_in_hex
from unittest import mock

def test_add_qubit_positions():
    # Create a mock graph
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])

    # Mock the get_qubit_coordinates function to return predetermined coordinates
    with mock.patch('src.soft_info.Hardware.coupling_map.get_qubit_coordinates', return_value=[(0, 0), (1, 0), (2, 0)]):
        add_qubit_positions(G, 3)

    # Check if node attributes are set correctly
    for qubit in G.nodes:
        assert 'pos' in G.nodes[qubit], "Position attribute should be set for each node"
        assert G.nodes[qubit]['pos'] == (qubit, 0), f"Incorrect position for qubit {qubit}"


@mock.patch('src.soft_info.Hardware.coupling_map.add_qubit_positions')
def test_create_coupling_graph_with_positions(mock_add_qubit_positions):
    # Create a mock backend configuration
    mock_configuration = mock.Mock()
    mock_configuration.coupling_map = [[0, 1], [1, 2], [2, 3]]
    mock_configuration.n_qubits = 4

    # Create a mock backend with the mock configuration
    backend_mock = mock.Mock()
    backend_mock.configuration.return_value = mock_configuration

    # Call the function
    G = create_coupling_graph_with_positions(backend_mock)

    # Check if the graph has the correct edges
    assert len(G.edges) == len(mock_configuration.coupling_map), "Graph should have the same number of edges as the coupling map"
    for edge in mock_configuration.coupling_map:
        assert G.has_edge(*edge), f"Graph should have an edge {edge}"

    # Check if add_qubit_positions was called correctly
    mock_add_qubit_positions.assert_called_once_with(G, mock_configuration.n_qubits)


# @mock.patch('src.soft_info.Hardware.coupling_map.create_coupling_graph_with_positions')
# @mock.patch('src.soft_info.Hardware.coupling_map.tqdm')
# def test_find_longest_path_general(mock_tqdm, mock_create_graph):
#     # Mock the graph creation
#     G = nx.Graph()
#     G.add_edges_from([(0, 1), (1, 2), (2, 3)])
#     print("Nodes:", G.nodes())
#     print("Edges:", G.edges())

#     mock_create_graph.return_value = G

#     # Mock backend
#     backend_mock = mock.Mock()

#     # Call the function
#     path, length, start_qubit = find_longest_path_general(backend_mock)

#     # Expected longest path in the mock graph
#     expected_path = [0, 1, 2, 3]
#     expected_length = 4
#     expected_start_qubit = 0

#     # Verify the result
#     assert path == expected_path, "The longest path is incorrect"
#     assert length == expected_length, "The length of the longest path is incorrect"
#     assert start_qubit == expected_start_qubit, "The starting qubit of the longest path is incorrect"



@mock.patch('src.soft_info.Hardware.coupling_map.create_coupling_graph_with_positions')
def test_find_longest_path_in_hex(mock_create_graph):
    # Mock the graph creation for a heavy hex graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])  # Last node (4) is a corner node
    mock_create_graph.return_value = G

    # Mock backend
    backend_mock = mock.Mock()

    # Call the function with a valid heavy hex map
    path, length, start_qubit = find_longest_path_in_hex(backend_mock)

    # Verify the result for a valid heavy hex map
    expected_path = [4, 3, 2, 1, 0]
    expected_length = 5
    expected_start_qubit = 4
    assert path == expected_path, "The longest path is incorrect"
    assert length == expected_length, "The length of the longest path is incorrect"
    assert start_qubit == expected_start_qubit, "The starting qubit of the longest path is incorrect"
