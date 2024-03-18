# 2023.11.30: Copied from https://github.com/qiskit-community/qopt-best-practices
"""Subset finders. Currently contains reference implementation
to evaluate 2-qubit gate fidelity."""

from __future__ import annotations
from dateutil import parser
import pytz

from typing import Dict, List, Tuple, Iterator

import numpy as np
from qiskit.providers import BackendV2
from qiskit.providers.models import BackendProperties
from rustworkx import EdgeList

class EvaluateBottlenecks:
    """Evaluates bottleneck for a given list of qubits based on the two-qubit gate error and readout error of specific qubits"""

    def __init__(self, backend: BackendV2, 
                 readout_multiplicator: float = 0.3, 
                 fidelity_multiplicator: float = 0.1,
                 date: str = None):
        props: BackendProperties

        if date is None:
            props = backend.properties()
        else:
            date = parser.parse(date)
            date = date.astimezone(pytz.utc)
            props = backend.properties(datetime=date)

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
        for edge in list(self.gate_errors.keys()):
            if edge[::-1] not in self.gate_errors:
                self.gate_errors[edge[::-1]] = self.gate_errors[edge]

        self.readout_errors: Dict[int, float] = {
            qubit: props.readout_error(qubit)
            for qubit in range(backend.configuration().n_qubits)
        }

        self.readout_multiplicator = readout_multiplicator
        self.fidelity_multiplicator = fidelity_multiplicator    

    def __call__(self, path: List[int]) -> float:
        if not path or len(path) == 1:
            return 0.0     

        max_gate_error = 0.0
        max_readout_error = 0.0
        fidelity = 1.0
        for idx, edge in enumerate(zip(path[0:-1], path[1:])):
            cx_error = self.gate_errors[edge]
            max_gate_error = max(max_gate_error, cx_error)

            if (idx+1) % 2 == 0: # only take the odd number (=ancilla) of qubits for the readout error
                readout_error = self.readout_errors[edge[0]] # we will miss the last qubit
                max_readout_error = max(max_readout_error, readout_error)
            
            fidelity *= 1 - cx_error

        # max_readout_error = max(max_readout_error, self.readout_errors[path[-1]]) # this is the last qubit NOT NEEDED BECAUSE CODE QUBIT
        # print(self.readout_multiplicator)
        return max_gate_error + self.readout_multiplicator * max_readout_error - self.fidelity_multiplicator*fidelity 
    
    def get_error_info(self, path: List[int]):
        if not path or len(path) == 1:
            return 0.0, 0.0
        
        gate_errors = []
        readout_errors = []
        ancilla_readout_errors = []
        fidelity = 1.0
        for idx, edge in enumerate(zip(path[0:-1], path[1:])):
            cx_error = self.gate_errors[edge]
            readout_error = self.readout_errors[edge[0]]
            gate_errors.append(cx_error)
            readout_errors.append(readout_error)
            if (idx+1) % 2 == 0:
                ancilla_readout_errors.append(readout_error)
            fidelity *= 1 - cx_error


        readout_errors.append(self.readout_errors[path[-1]])

        return (np.mean(gate_errors), np.min(gate_errors), np.max(gate_errors), 
                np.mean(readout_errors), np.min(readout_errors), np.max(readout_errors), 
                np.mean(ancilla_readout_errors), np.min(ancilla_readout_errors), np.max(ancilla_readout_errors), 
                fidelity)

        



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

        for edge in list(self.gate_errors.keys()):
            if edge[::-1] not in self.gate_errors:
                self.gate_errors[edge[::-1]] = self.gate_errors[edge]

    def __call__(self, path: List[int]) -> float:
        if not path or len(path) == 1:
            return 0.0

        fidelity = 1.0
        for edge in zip(path[0:-1], path[1:]): # takes the first and second element of the path, then the second and third, and so on
            cx_error = self.gate_errors[edge]
            # try:
            #     cx_error = self.gate_errors[edge]

            # except:  # pylint: disable=bare-except
            #     # This handles the reverse case
            #     cx_error = self.gate_errors[edge[::-1]]
            fidelity *= 1 - cx_error

        return fidelity
    
    
    # def pairwise(self, iterable: Iterator[int]) -> Iterator[Tuple[int, int]]:
    #     "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    #     a, b = tee(iterable)
    #     next(b, None)
    #     return zip(a, b)

    # def __call__(self, path_iter: Iterator[int], edges: EdgeList) -> float:
    #     if path_iter is None:
    #         return 0.0

    #     fidelity = 1.0
    #     for edge in self.pairwise(path_iter):
    #         cx_error = self.gate_errors[edge]

    #         fidelity *= 1 - cx_error

    #     return fidelity



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
