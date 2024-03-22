# Maurice Hanisch mhanisc@ethz.ch
# Created 2024-03-14

from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2

from Scratch import metadata_helper


def run_IQ_calibration(backend: BackendV2, shots: int = None) -> None:
    n_qubits = backend.configuration().n_qubits
    transpiled_circuits = {}

    if shots != None:
        assert shots*n_qubits < 5e6, "Too many shots, expect a PAYLOAD error."
    else:
        shots = int(1e6//n_qubits)

    # 0 state
    qc_0 = QuantumCircuit(n_qubits, n_qubits)
    qc_0.measure(range(n_qubits), range(n_qubits))
    transpiled_circuits["transpile_qc_0"] = transpile(qc_0, backend, optimization_level=0)

    # 1 state
    qc_1 = QuantumCircuit(n_qubits, n_qubits)
    qc_1.x(range(n_qubits))
    qc_1.measure(range(n_qubits), range(n_qubits))
    transpiled_circuits["transpile_qc_1"] = transpile(qc_1, backend, optimization_level=0)

    # Double measurements
    qcd_0 = QuantumCircuit(n_qubits, 2*n_qubits)
    qcd_0.measure(range(n_qubits), range(n_qubits))
    qcd_0.measure(range(n_qubits), range(n_qubits, 2*n_qubits))

    qcd_1 = QuantumCircuit(n_qubits, 2*n_qubits)
    qcd_1.x(range(n_qubits))
    qcd_1.measure(range(n_qubits), range(n_qubits))
    qcd_1.measure(range(n_qubits), range(n_qubits, 2*n_qubits))

    transpiled_circuits["transpile_qc_0_double"] = transpile(qcd_0, backend, optimization_level=0)
    transpiled_circuits["transpile_qc_1_double"] = transpile(qcd_1, backend, optimization_level=0)


    for i in [0, 1]:
        metadata = metadata_helper(num_qubits=n_qubits, sampled_state=f"{i}"*n_qubits, 
                                   optimization_level=0)
        backend.run(metadata, transpiled_circuits[f"transpile_qc_{i}"], shots=shots, 
                    meas_level=1, meas_return='single', job_tags=[f"Calibration, shots {shots}"])

        metadata = metadata_helper(num_qubits=n_qubits, sampled_state=f"{i}"*n_qubits, 
                                   optimization_level=0, double_msmt=True)
        backend.run(metadata, transpiled_circuits[f"transpile_qc_{i}_double"], shots=shots/2, 
                    meas_level=1, meas_return='single', job_tags=[f"Calibration, shots {shots}", "Double_Measurement"])



