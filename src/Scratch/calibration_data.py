# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-27

import warnings
import re
from typing import List, Optional

import pandas as pd

from .metadata import metadata_loader


def extract_backend_name(backend_str):
    match = re.search(r"IBMBackend\('([^']+)'\)", backend_str)
    if match:
        # Return the extracted backend name
        return match.group(1)
    else:
        # Return None or raise an error if the pattern is not found
        return None

def load_calibration_memory(provider, device: Optional[str] = None, qubits: Optional[List[int]] = None, tobecalib_job: Optional[str] = None, _take_newest=True):
    """
    Loads the calibration memory for the given device and qubits.

    Args:
        provider: IBMQ provider
        device: String of the device name
        qubits: List of qubits to load calibration memory for
        _take_newest: If True, only the two newest calibration jobs are loaded
        to_be_calibrated_job_id: Optional job ID for specific calibration data

    Returns:
        A dictionary with the calibration memory for each qubit
    """
    if not device and not tobecalib_job:
        raise ValueError("Either 'device' or 'to_be_calibrated_job_id' must be provided.")
    
    if not qubits:
        qubits = list(range(127)) # Hardcoded for biggest device TODO: retrieve the num qubits from the device
        warnings.warn("No qubits specified, loading calibration data for all qubits.")
        
    specified_job_creation_date = None
    if tobecalib_job:
        specified_job = provider.retrieve_job(tobecalib_job)
        specified_job_creation_date = pd.to_datetime(specified_job.creation_date())
        backend_name = extract_backend_name(specified_job.backend())

        if device and device != backend_name:
            raise ValueError(f"The specified job's backend: {specified_job.backend()} does not match the provided backend name: {device}.")
        device = backend_name
        # print(device)
   
    md = metadata_loader(_extract=True, _drop_inutile=True).dropna(subset=["num_qubits"])
    # Filter metadata
    mask = (
        (md["backend_name"] == device) &
        (md["job_status"] == "JobStatus.DONE") &
        (md["optimization_level"] == 0)
    )
    md_filtered = md.loc[mask]

    all_memories = {qubit: {} for qubit in qubits}

    for state in ['0', '1']:
        state_mask = md_filtered["sampled_state"] == md_filtered["num_qubits"].apply(lambda x: state * int(x))
        md_state_filtered = md_filtered[state_mask].copy()

        if specified_job_creation_date is not None:
            md_state_filtered['creation_date'] = pd.to_datetime(md_state_filtered['creation_date'])
            closest_job_id = md_state_filtered.iloc[(md_state_filtered['creation_date'] - specified_job_creation_date).abs().argsort()[:1]]['job_id'].values[0]
            job_ids = [closest_job_id]
        elif _take_newest:
            job_ids = md_state_filtered["job_id"][:1]
        else:
            job_ids = md_state_filtered["job_id"]
            warnings.warn(f"Loading multiple calibration jobs frecusively for state {state}.")

        for job_id in job_ids:
            mmr_name = f"mmr_{state}"
            job = provider.retrieve_job(job_id)
            memory = job.result().get_memory()

            for qubit in qubits:
                if qubit < memory.shape[1]:  # Check if qubit index is valid
                    all_memories[qubit][mmr_name] = memory[:, int(qubit)]

    for qubit, memories in all_memories.items():
        if len(memories) != 2:
            warnings.warn(f"Loaded {len(memories)} memories with keys {list(memories.keys())}, expected 2 for qubit {qubit}.")

    return all_memories





