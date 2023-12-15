# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-27

import warnings
import re
from typing import List, Optional

import pandas as pd
import json

from ..metadata import metadata_loader, find_and_create_scratch


def extract_backend_name(backend_str):
    match = re.search(r"IBMBackend\('([^']+)'\)", backend_str)
    if match:
        # Return the extracted backend name
        return match.group(1)
    else:
        # Return None or raise an error if the pattern is not found
        return None
    

def find_closest_calib_jobs(tobecalib_job: str, other_date = None):
    """Find the closest calibration jobs for the given job ID."""
    # Find attributes of the tobecalib_job
    root_dir = find_and_create_scratch()
    metadata_path = f"{root_dir}/job_metadata.json"
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    specified_job_entry = next((item for item in metadata if item['job_id'] == tobecalib_job), None)
    if not specified_job_entry:
        raise ValueError(f"No job found in metadata with ID: {tobecalib_job}")

    specified_job_creation_date = pd.to_datetime(specified_job_entry['creation_date'])
    specified_job_creation_date = specified_job_creation_date.tz_convert('UTC') if specified_job_creation_date.tzinfo else specified_job_creation_date


    specified_job_execution_date = pd.to_datetime(specified_job_entry['additional_metadata']['execution_date'])
    print(f"Specified job execution date: {specified_job_execution_date}")
    needed_calib_date = specified_job_execution_date

    if other_date is not None: # if specified, use other date as closest date
        needed_calib_date = pd.to_datetime(other_date, utc=True)

    backend_name = specified_job_entry['backend_name']

    # Find the calibration job ID using the metadata
    md = metadata_loader(_extract=True, _drop_inutile=True).dropna(subset=["num_qubits"])
    mask = (
        (md["backend_name"] == backend_name) &
        (md["job_status"] == "JobStatus.DONE") &
        (md["optimization_level"] == 0)
    )
    md_filtered = md.loc[mask]
    
    job_ids = {}
    creation_dates = {}
    execution_dates = {}
    for state in ['0', '1']:
        state_mask = md_filtered["sampled_state"] == md_filtered["num_qubits"].apply(lambda x: state * int(x))
        md_state_filtered = md_filtered[state_mask].copy()
        md_state_filtered['execution_date'] = pd.to_datetime(md_state_filtered['execution_date'], utc=True, format='ISO8601')
        closest_job_info = md_state_filtered.iloc[(md_state_filtered['execution_date'] - needed_calib_date).abs().argsort()[:1]]
        closest_job_id = closest_job_info['job_id'].values[0]

        closest_creation_date_np = closest_job_info['creation_date'].values[0]
        closest_creation_date = pd.to_datetime(closest_creation_date_np, utc=True).to_pydatetime()
        creation_dates[state] = closest_creation_date

        closest_execution_date_np = closest_job_info['execution_date'].values[0]
        closest_execution_date = pd.to_datetime(closest_execution_date_np, utc=True).to_pydatetime()
        execution_dates[state] = closest_execution_date

        job_ids[state] = closest_job_id
    
    # Check if year, day, and hour are the same for both states
    date_0 = execution_dates['0']
    date_1 = execution_dates['1']

    if not (date_0.year == date_1.year and date_0.day == date_1.day and date_0.hour == date_1.hour):
        raise ValueError("Year, day, and hour of creation dates for the closest jobs are different for each state.")

    print(f"Found jobs for backend {backend_name} with closest execution date {execution_dates['0']}.")
    return job_ids, backend_name, creation_dates['0']



def load_calibration_memory(provider, tobecalib_job: str, qubits: Optional[List[int]] = None, other_date = None): 
    """Load the calibration memory for the closest calibration jobs for the given job ID."""
    if not tobecalib_job:
        raise NotImplementedError("Only loading calibration data for a specific job is currently supported.")
    
    closest_job_ids, _, _ = find_closest_calib_jobs(tobecalib_job, other_date=other_date)

    all_memories = {}
    for state, job_id in closest_job_ids.items():
        mmr_name = f"mmr_{state}"
        job = provider.retrieve_job(job_id)
        memory = job.result().get_memory()

        if qubits is None:
            qubits = range(memory.shape[1])  # Assuming all qubits are included

        for qubit in qubits:
            if qubit < memory.shape[1]:  # Check if qubit index is valid
                if qubit not in all_memories:
                    all_memories[qubit] = {}
                all_memories[qubit][mmr_name] = memory[:, int(qubit)]

    return all_memories


def load_calibration_memory_old(provider, device: Optional[str] = None, qubits: Optional[List[int]] = None, tobecalib_job: Optional[str] = None, _take_newest=True):
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
    
    specified_job_creation_date = None
    if tobecalib_job:
        root_dir = find_and_create_scratch()
        metadata_path = f"{root_dir}/job_metadata.json"
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)

        specified_job_entry = next((item for item in metadata if item['job_id'] == tobecalib_job), None)

        if not specified_job_entry:
            raise ValueError(f"No job found in metadata with ID: {tobecalib_job}")

        # Extract creation date and backend name
        specified_job_creation_date = pd.to_datetime(specified_job_entry['creation_date'])
        backend_name = specified_job_entry['backend_name']
        if device and device != backend_name:
            raise ValueError(f"The specified job's backend: {backend_name} does not match the provided backend name: {device}.")
        device = backend_name
   
    md = metadata_loader(_extract=True, _drop_inutile=True).dropna(subset=["num_qubits"])
    # Filter metadata
    mask = (
        (md["backend_name"] == device) &
        (md["job_status"] == "JobStatus.DONE") &
        (md["optimization_level"] == 0)
    )
    md_filtered = md.loc[mask]

    _return_qubits = False
    if qubits is None:
        _return_qubits = True
        num_qubits = md_filtered["num_qubits"].unique()
        if len(num_qubits) != 1:
            raise ValueError(f"Multiple number of qubits found: {num_qubits}")
        qubits = list(range(int(num_qubits[0])))


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
            warnings.warn(f"Loading multiple calibration jobs recusively for state {state}.")

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

    if _return_qubits:
        return (all_memories, qubits) # TODO: is there a better way to do this?
    
    return all_memories





