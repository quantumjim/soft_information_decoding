# 2023.11.30: Copied from https://github.com/qiskit-community/qopt-best-practices
"""Backend Evaluator"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, List, Tuple

import numpy as np
from qiskit.providers import BackendV2
from qiskit.transpiler import CouplingMap
from rustworkx import EdgeList

from .metric_evaluators import EvaluateFidelity, GateLengths
from .qubit_subset_finders import find_lines

from ..coupling_map import create_coupling_graph_with_positions, highlight_path


class BackendEvaluator:
    """
    Finds best subset of qubits for a given device that maximizes a given
    metric for a given geometry.
    This subset can be provided as an initial_layout for the SwapStrategy
    transpiler pass.
    """

    def __init__(self, backend: BackendV2, symmetric: bool = False):
        """Create a backend evaluator which will determine the best set of qubits to use.

        Args:
            backend: The backend to evaluate.
            symmetric: Whether to treat the coupling map as symmetric. If False, the coupling map is
                taken as directional, as reported by the backend. Defaults to False.
        """
        self.backend = backend
        self.coupling_map = CouplingMap(backend.configuration().coupling_map)
        if symmetric:
            self.coupling_map.make_symmetric()

    @classmethod
    def __metadata_eval(
        cls, chain: List[int], gate_lengths: GateLengths, edges: EdgeList
    ) -> Dict[str, Any]:
        _min, _max, _mean = gate_lengths(chain, edges)
        return {"min": _min, "max": _max, "mean": _mean, "diff": _max - _min}

    def __get_valid_chains(
        self,
        num_qubits: int,
        subset_finder: (
            Callable[[int, BackendV2, CouplingMap], List[List[int]]] | None
        ) = None,
    ) -> List[List[int]]:
        if subset_finder is None:
            subset_finder = find_lines

        # TODO: add callbacks
        qubit_subsets = subset_finder(num_qubits, self.backend, self.coupling_map)
        return qubit_subsets

    def top_N(
        self,
        num_qubits: int,
        N: int,
        subset_finder: (
            Callable[[int, BackendV2, CouplingMap], List[List[int]]] | None
        ) = None,
        metric_eval: Callable[[List[int], EdgeList], float] | None = None,
        metadata_eval: Callable[[List[int], EdgeList], Dict[str, Any]] | None = None,
    ) -> List[Tuple[List[int] | None, Any, Dict[str, Any]]]:
        if metric_eval is None:
            metric_eval = EvaluateFidelity(self.backend)
        if metadata_eval is None:
            gate_lengths = GateLengths(self.backend)

            def __metadata_eval(subset: List[int], edges: EdgeList) -> Dict:
                return self.__metadata_eval(subset, gate_lengths, edges)

            metadata_eval = __metadata_eval

        qubit_subsets = self.__get_valid_chains(
            num_qubits=num_qubits, subset_finder=subset_finder
        )

        if len(qubit_subsets) == 0:
            # No valid subsets
            return []

        # evaluating the subsets
        edges = self.coupling_map.get_edges()
        scores: List[float]
        scores = [metric_eval(subset, edges) for subset in qubit_subsets]

        i_top_N: np.ndarray
        i_top_N = np.argpartition(scores, len(scores) - N)[-N:]
        top_N_subsets = [qubit_subsets[i] for i in i_top_N]
        top_N_scores = [scores[i] for i in i_top_N]

        # Return the best subset sorted by score

        return [
            (
                inst_subset,
                inst_score,
                metadata_eval(inst_subset, self.coupling_map),
            )
            for inst_subset, inst_score in zip(top_N_subsets, top_N_scores)
        ]

    def evaluate(
        self,
        num_qubits: int,
        subset_finder: (
            Callable[[int, BackendV2, CouplingMap], List[List[int]]] | None
        ) = None,
        metric_eval: Callable[[List[int], EdgeList], Any] | None = None,
        metadata_eval: Callable[[List[int], EdgeList], Dict[str, Any]] | None = None,
        plot: bool = False,
    ) -> Tuple[List[int] | None, Any, int, Dict[str, Any]]:
        """
        Args:
            num_qubits: the number of qubits
            subset_finder: callable, will default to "find_line"
            metric_eval: callable, will default to "EvaluateFidelity"
            metadata_eval: callable, will default to "self.__metadata_eval". Computes metadata
                related to the best subset.

        Returns:
            The tuple ``(qubits, score, n_subsets, metadata)`` containing the best qubits for the
            given metric, the metric for said qubits, and the number of subsets evaluated. If no
            subsets were found, then ``qubits`` is ``None``.
        """

        if metric_eval is None:
            metric_eval = EvaluateFidelity(self.backend)
        if metadata_eval is None:
            gate_lengths = GateLengths(self.backend)

            def __metadata_eval(subset: List[int], edges: EdgeList) -> Dict:
                return self.__metadata_eval(subset, gate_lengths, edges)

            metadata_eval = __metadata_eval

        qubit_subsets = self.__get_valid_chains(
            num_qubits=num_qubits, subset_finder=subset_finder
        )

        if len(qubit_subsets) == 0:
            # No valid subsets
            return None, -1, 0, {}

        # evaluating the subsets
        edges = self.coupling_map.get_edges()
        scores = [metric_eval(subset, edges) for subset in qubit_subsets]

        # Return the best subset sorted by score
        best_subset, best_score = min(zip(qubit_subsets, scores), key=lambda x: -x[1])
        best_metadata = metadata_eval(best_subset, self.coupling_map)
        num_subsets = len(qubit_subsets)

        if plot:
            G = create_coupling_graph_with_positions(self.backend)
            highlight_path(G, best_subset)

        return best_subset, best_score, num_subsets, best_metadata
    
    

