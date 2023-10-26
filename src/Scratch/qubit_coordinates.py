# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-26

import json
from .core import find_and_create_scratch


def get_qubit_coordinates(n_qubits, rotated=True):
    """
    Retrieves the qubit coordinates from a JSON file and optionally rotates them.

    Parameters:
    - n_qubits (int): The number of qubits for which to retrieve coordinates.
    - rotated (bool, optional): Whether to rotate the coordinates counterclockwise by 90 degrees. 
                                Defaults to True.

    Returns:
    - list of tuple: A list of tuples containing the (x, y) coordinates of each qubit. 
                     The coordinates are rotated if `rotated` is True.

    Example:
    >>> get_qubit_coordinates(5, rotated=True)
    [(y1, -x1), (y2, -x2), ...]

    """
    root_dir = find_and_create_scratch()

    with open(f"{root_dir}/qubit_coordinates.json", "r") as f:
        qubit_coordinates_full = json.load(f)
    
    qubit_coordinates = qubit_coordinates_full[str(n_qubits)]
    
    if rotated:
        qubit_coordinates = [(y, -x) for (x, y) in qubit_coordinates]

    return qubit_coordinates