# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-27

import warnings

from .metadata import metadata_loader

def load_calibration_memory(provider, device, qubit, _take_newest=True):
    md = metadata_loader(extract=True).dropna(subset=["num_qubits"])

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

    memories = {}
    for job_id, sampled_state in zip(md_filtered["job_id"], md_filtered["sampled_state"]):
        mmr_name = f"mmr_{sampled_state[0]}"
        job = provider.retrieve_job(job_id)
        memory = job.result().get_memory()

        if qubit < memory.shape[1]:  # Check if qubit index is valid
            memories[mmr_name] = memory[:, int(qubit)]

    if len(memories) != 2:
        warnings.warn(
            f"Loaded {len(memories)} memories with keys {list(memories.keys())}, expected 2.")

    return memories
