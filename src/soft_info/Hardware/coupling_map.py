# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-26

import warnings

import networkx as nx
from networkx import set_node_attributes
import matplotlib.pyplot as plt
from tqdm import tqdm

from Scratch import get_qubit_coordinates


def add_qubit_positions(G, n_qubits: int):
    """Add position attributes to the graph nodes."""

    qubit_coordinates = get_qubit_coordinates(n_qubits)
    pos_dict = {}
    for qubit, coordinates in enumerate(qubit_coordinates):
        pos_dict[qubit] = coordinates
    set_node_attributes(G, pos_dict, 'pos')


def create_coupling_graph_with_positions(backend):
    """Create a graph from the coupling map and add qubit positions."""
    coupling_map = backend.configuration().coupling_map
    n_qubits = backend.configuration().n_qubits
    G = nx.Graph()
    for edge in coupling_map:
        G.add_edge(edge[0], edge[1])
    add_qubit_positions(G, n_qubits)
    return G


def highlight_path(G, path):
    """Highlight the given path in the graph.

    Args: 
        G: Graph to highlight the path in.
        path: List of nodes in the path.
    """
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        warnings.warn("No positions found for the nodes of the input graph.")
    nx.draw(G, pos, with_labels=True, edge_color='white')
    path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]

    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color='red', width=4)
    plt.show()


def find_longest_path_dfs(G, start, visited, path, longest_path_info):
    visited.add(start)
    path.append(start)

    if len(path) > longest_path_info[0]:
        longest_path_info[0] = len(path)
        longest_path_info[1] = path.copy()
        # Record the start of this new longest path
        if longest_path_info[2] is None or len(path) > longest_path_info[0]:
            longest_path_info[2] = path[0]

    for neighbor in G.neighbors(start):
        if neighbor not in visited:
            find_longest_path_dfs(G, neighbor, visited,
                                  path, longest_path_info)

    visited.remove(start)
    path.pop()



def find_longest_path_general(backend, plot=False):
    """
    Find the longest simple path in a graph generated from a quantum backend's coupling map.

    Parameters:
        backend (obj): A quantum backend object with a configuration attribute that contains the coupling map.
        plot (bool, optional): If True, the longest path will be highlighted on the graph plot. Defaults to False.

    Returns: 
        tuple: (path, length, start_qubit)
        A tuple containing the longest path as a list of qubits, the length of the longest path, and the starting qubit.
    """
    G = create_coupling_graph_with_positions(backend)

    # Length of longest path, longest path, starting qubit
    longest_path_info = [0, [], None]

    for start_qubit in tqdm(G.nodes(), desc=f"Finding the longest path starting from {len(G.nodes())} qubits"):
        visited = set()
        find_longest_path_dfs(G, start_qubit, visited, [], longest_path_info)

    length, path, start_qubit = longest_path_info

    if plot:
        highlight_path(G, path)

    return path, length, start_qubit


def find_longest_path_in_hex(backend, plot=False):
    """
    Find the longest simple path in a heavy hex graph generated from a quantum backend's coupling map.

    Parameters:
        backend (obj): A quantum backend object with a configuration attribute that contains the coupling map.
        plot (bool, optional): If True, the longest path will be highlighted on the graph plot. Defaults to False.

    Returns:
        tuple or str: (path, length, start_qubit)
        A tuple containing the longest path as a list of qubits, the length of the longest path, and the starting qubit.
        If the graph is not a valid heavy hex map, returns a string describing the issue.
    """
    G = create_coupling_graph_with_positions(backend)

    corner_nodes = [node for node in G.nodes() if G.degree(node) == 1]

    if len(corner_nodes) != 2:
        warnings.warn(
            (f"Not a valid heavy hex map, found {len(corner_nodes)} corner" 
             +"nodes instead of 2. Looking for longest path starting with the"
             +" last corner node."))

    visited = set()
    longest_path_info = [0, [], None]
    find_longest_path_dfs(G, corner_nodes[-1], visited, [], longest_path_info)

    length, path, start_qubit = longest_path_info

    if plot:
        highlight_path(G, path)

    return path, length, start_qubit
