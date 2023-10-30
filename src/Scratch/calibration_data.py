# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-27

import warnings
from typing import List

from .metadata import metadata_loader

def load_calibration_memory(provider, device, qubits: List[int], _take_newest=True):
    """Loads the calibration memory for the given device and qubits.

    Args:
        provider: IBMQ provider
        device: String of the device name
        qubits: List of qubits to load calibration memory for
        _take_newest: If True, only the two newest calibration jobs are loaded

    Returns:
        A dictionary with the calibration memory for each qubit
    """
    md = metadata_loader(_extract=True).dropna(subset=["num_qubits"])

    mask = (
        (md["backend_name"] == device) &
        (md["job_status"] == "JobStatus.DONE") &
        (md["optimization_level"] == 0) &
        (
            (md["sampled_state"] == md["num_qubits"].apply(lambda x: '1' * int(x))) |
            (md["sampled_state"] == md["num_qubits"].apply(lambda x: '0' * int(x)))
        )
    )
    md_filtered = md.loc[mask]
    if _take_newest:
        md_filtered = md_filtered[:2]  # Only take newest two jobs

    all_memories = {qubit: {} for qubit in qubits}

    for job_id, sampled_state in zip(md_filtered["job_id"], md_filtered["sampled_state"]):
        mmr_name = f"mmr_{sampled_state[0]}"
        job = provider.retrieve_job(job_id)
        memory = job.result().get_memory()

        for qubit in qubits:
            if qubit < memory.shape[1]:  # Check if qubit index is valid
                all_memories[qubit][mmr_name] = memory[:, int(qubit)]

    for qubit, memories in all_memories.items():
        if len(memories) != 2:
            warnings.warn(
                f"Loaded {len(memories)} memories with keys {list(memories.keys())}, expected 2 for qubit {qubit}.")

    return all_memories





