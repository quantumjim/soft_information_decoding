# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-26

import warnings

from .coupling_map import find_longest_path_in_hex, find_longest_path_general


def get_repcode_layout(distance, backend, _is_hex=True, plot=False):
    """Get the layout of repetition code based on the longest path and distance.

    Args:
        distance (int): Maximum number of code qubits.
        backend: Backend object providing coupling map.
        _is_hex (bool): If True, the layout is found in a hexagonal topology; otherwise, it's general.
        plot (bool): If True, the longest path is plotted.

    Returns:
        list: The layout consisting of code qubits followed by link qubits.
    """
    find_path = find_longest_path_in_hex if _is_hex else find_longest_path_general
    path, _, _ = find_path(backend, plot=plot)

    if 2 * distance - 1 > len(path):
        raise ValueError(
            f"The distance: {distance} is larger than the max distance: "
            f"{int((len(path) + 1) / 2)} for the given path of length: {len(path)}"
        )

    bounded_path = path[:2 * distance - 1]
    layout = bounded_path[1::2] + bounded_path[::2]
    return layout
