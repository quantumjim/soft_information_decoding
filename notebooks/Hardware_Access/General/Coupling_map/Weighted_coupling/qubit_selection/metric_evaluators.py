# 2023.11.30: Copied from https://github.com/qiskit-community/qopt-best-practices
"""Subset finders. Currently contains reference implementation
to evaluate 2-qubit gate fidelity."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from qiskit.providers import BackendV2
from qiskit.providers.models import BackendProperties
from rustworkx import EdgeList


class EvaluateFidelity:
    """Evaluates fidelity on a given list of qubits based on the two-qubit gate error
    for a specific backend.
    """

    def __init__(self, backend: BackendV2):
        props: BackendProperties
        props = backend.properties()

        if "cx" in backend.configuration().basis_gates:
            gate_name = "cx"
        elif "ecr" in backend.configuration().basis_gates:
            gate_name = "ecr"
        elif "cz" in backend.configuration().basis_gates:
            gate_name = "cz"
        else:
            raise ValueError("Could not identify two-qubit gate")

        self.gate_errors: Dict[Tuple[int, ...], float] = {
            _edge: metadata.get("gate_error", [-1])[0]
            for _edge, metadata in props.gate_property(gate_name).items()
        }

    def __call__(self, path: List[int], edges: EdgeList) -> float:
        if not path or len(path) == 1:
            return 0.0

        fidelity = 1.0
        for edge in zip(path[0:-1], path[1:]):
            try:
                cx_error = self.gate_errors[edge]

            except:  # pylint: disable=bare-except
                # This handles the reverse case
                cx_error = self.gate_errors[edge[::-1]]

            fidelity *= 1 - cx_error

        return fidelity


class GateLengths:
    """Evaluates max, min, and mean gate lengths for a given list of qubits.

    Returns:
       Tuple ``(min, max, mean)`` containing the minimum, maximum, and mean two-qubit gate durations
        for the given list of qubits.
    """

    def __init__(self, backend: BackendV2) -> None:
        props: BackendProperties
        props = backend.properties()

        if "cx" in backend.configuration().basis_gates:
            gate_name = "cx"
        elif "ecr" in backend.configuration().basis_gates:
            gate_name = "ecr"
        elif "cz" in backend.configuration().basis_gates:
            gate_name = "cz"
        else:
            raise ValueError("Could not identify two-qubit gate")

        self.gate_durations: Dict[Tuple[int, ...], float] = {
            _edge: metadata.get("gate_length", (np.inf, None))[0]
            for _edge, metadata in props.gate_property(gate_name).items()
        }

    def __call__(self, path: List[int], edges: EdgeList) -> Tuple[float, float, float]:
        two_qubit_durations: Dict[Tuple[int, ...], float] = {}

        for edge in edges:
            try:
                gate_dur = self.gate_durations[edge]

            except:  # pylint: disable=bare-except
                # This handles the reverse case
                gate_dur = self.gate_durations[edge[::-1]]

            two_qubit_durations[tuple(edge)] = gate_dur

        if not path or len(path) == 1:
            return (np.nan, np.nan, np.nan)

        return (
            np.min(list(two_qubit_durations.values())),
            np.max(list(two_qubit_durations.values())),
            np.mean(list(two_qubit_durations.values())),
        )
