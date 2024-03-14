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

    paths = [list(c) for a in all_paths for b in a for c in a[b]]
    return paths

    # # TODO This list comprehension is slow. Improve its performance.
    # paths = np.asarray(
    #     [
    #         # TODO Increase dtype to uint8 once running on devices with more than 256 qubits.
    #         np.array((list(c), list(sorted(list(c)))), dtype=np.uint8)
    #         for a in iter(all_paths)
    #         for b in iter(a)
    #         for c in iter(a[b])
    #     ]
    # )

    # if len(paths) == 0:
    #     return []

    # # filter out duplicated paths
    # _, unique_indices = np.unique(paths[:, 1], return_index=True, axis=0)

    # filtered_paths = paths[:, 0][unique_indices].tolist()

    # return filtered_paths

    # Directly compile list of paths without checking for uniqueness

# def find_lines(length: int, backend: BackendV2, coupling_map: CouplingMap | None = None) -> List[List[int]]:
#     if coupling_map is None:
#         coupling_map = CouplingMap(backend.configuration().coupling_map)

#     all_paths_iter = rx.all_pairs_all_simple_paths(
#         coupling_map.graph,
#         min_depth=length,
#         cutoff=length,
#     ).values()

#     # Function to flatten and listify paths from the iterator
#     def listify_paths(a):
#         return [list(c) for b in a for c in a[b]]

#     # Use ThreadPoolExecutor to parallelize the listify operation
#     paths = []
#     with ThreadPoolExecutor() as executor:
#         # Submit tasks
#         futures = [executor.submit(listify_paths, a) for a in all_paths_iter]

#         # Wait for tasks to complete and collect results
#         for future in as_completed(futures):
#             paths.extend(future.result())

#     return paths







# def find_lines(
#     length: int,
#     backend: BackendV2,
#     coupling_map: CouplingMap | None = None,
# ) -> List[List[int]]:
#     """Finds all possible lines of length `length` for a specific backend topology.

#     This method can take quite some time to run on large devices since there
#     are many paths.

#     Returns:
#         The found paths.
#     """

#     # might make sense to make backend the only input for simplicity
#     if coupling_map is None:
#         coupling_map = CouplingMap(backend.configuration().coupling_map)

#     all_paths = rx.all_pairs_all_simple_paths(
#         coupling_map.graph,
#         min_depth=length,
#         cutoff=length,
#     )

#     return all_paths.values()
