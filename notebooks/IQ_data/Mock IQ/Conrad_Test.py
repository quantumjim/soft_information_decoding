# Checking multiple qubits with MockIQBackend
import numpy as np
from qiskit_experiments.test import MockIQBackend, MockIQExperimentHelper
from qiskit_experiments.test.mock_iq_helpers import MockIQRabiHelper
from qiskit import QuantumCircuit
import itertools as it
from typing import List, Dict, Any

N = 14
circ = QuantumCircuit(N, N)


class MyHelper(MockIQExperimentHelper):
    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, Any]]:
        output = []
        circ: QuantumCircuit
        for circ in circuits:
            probs = {
                "".join(s): 1 if "1" not in s else 0
                for s in it.product("01", repeat=circ.num_qubits)
            }
            output.append(probs)
        return output


backend = MockIQBackend(
    experiment_helper=MyHelper(
        iq_cluster_centers=[((-1.0, -1.0), (1.0, 1.0))
                            for _ in range(circ.num_qubits)]
    )
)
job = backend.run([circ])


print(np.array(job.result().results[0].data.memory).shape)
