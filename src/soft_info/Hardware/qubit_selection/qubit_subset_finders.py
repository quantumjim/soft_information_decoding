# 2023.11.30: Copied from https://github.com/qiskit-community/qopt-best-practices
"""Subset finders. Currently contains reference implementation
to find lines."""

from __future__ import annotations

import numpy as np
import rustworkx as rx
from qiskit.transpiler import CouplingMap
from typing import List
from qiskit.providers import BackendV2


# TODO: backend typehint. Currently, only BackendV1 is supported
#       Might make sense to extend to BackendV2 for generality
def find_lines(
    length: int,
    backend: BackendV2,
    coupling_map: CouplingMap | None = None,
) -> List[List[int]]:
    """Finds all possible lines of length `length` for a specific backend topology.

    This method can take quite some time to run on large devices since there
    are many paths.

    Returns:
        The found paths.
    """

    # might make sense to make backend the only input for simplicity
    if coupling_map is None:
        coupling_map = CouplingMap(backend.configuration().coupling_map)

    all_paths = rx.all_pairs_all_simple_paths(
        coupling_map.graph,
        min_depth=length,
        cutoff=length,
    ).values()

    # TODO This list comprehension is slow. Improve its performance.
    paths = np.asarray(
        [
            # TODO Increase dtype to uint8 once running on devices with more than 256 qubits.
            np.array((list(c), list(sorted(list(c)))), dtype=np.uint8)
            for a in iter(all_paths)
            for b in iter(a)
            for c in iter(a[b])
        ]
    )

    if len(paths) == 0:
        return []

    # filter out duplicated paths
    _, unique_indices = np.unique(paths[:, 1], return_index=True, axis=0)

    filtered_paths = paths[:, 0][unique_indices].tolist()

    return filtered_paths
