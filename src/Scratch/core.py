# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-20

import json
import os
import glob
import inspect
import subprocess

from datetime import datetime

import numpy as np

from qiskit.providers import JobStatus


def spawn_subprocess(job_id, provider, additional_dict_str, root_dir):
    lock_file = f'{root_dir}/tmp/{job_id}.lock'
    if os.path.exists(lock_file):
        print(f'Subprocess for job {job_id} already exists.')
        return
    # Create a lock file
    open(lock_file, 'a').close()
    subprocess.Popen(['python', 'monitor_job.py', job_id, provider, additional_dict_str], preexec_fn=lambda: os.remove(lock_file))



def find_and_create_scratch():
    original_path = os.getcwd()
    scratch_path = None
    while True:
        current_folder = os.path.basename(os.getcwd())
        if current_folder == 'Soft-Info':
            scratch_path = os.path.join(os.getcwd(), '.Scratch')
            if not os.path.exists('.Scratch'):
                os.mkdir('.Scratch')
            break
        else:
            os.chdir('..')
            if os.getcwd() == '/':  # Stop if we reach the root directory
                print("Soft-Info folder not found.")
                break
    os.chdir(original_path)  # Navigate back to original directory
    return scratch_path


def create_unique_filename(json_filename, day_dir):
    i = 2
    new_filename = json_filename
    filename_without_extension = os.path.splitext(json_filename)[0]
    extension = os.path.splitext(json_filename)[1]

    while os.path.exists(f"{day_dir}/{new_filename}"):
        new_filename = f"{filename_without_extension}v{i}{extension}"
        i += 1
    return new_filename


def make_serializable(obj):
    if isinstance(obj, list):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    else:
        return obj


def update_metadata(root_dir, year_str, month_str, day_str, time_str, json_path, job_id, notebook_name,
                    memory_shape, memory_type, _is_complex, time_taken, additional_dict=None):
    '''Updates the metadata file with possible additional dicts.'''
    print("Updating metadata...")
    metadata_file_path = os.path.join(root_dir, f"{month_str}_metadata.json")

    # Load or create the metadata dictionary
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, 'r') as f:
            metadata_dict = json.load(f)
    else:
        metadata_dict = {}

    # Create new metadata entry
    metadata_entry = {
        'job_id': job_id,
        'notebook_name': notebook_name,
        'memory_shape': memory_shape,
        'memory_type': memory_type,
        '_is_complex': _is_complex,
        'time_taken': time_taken,
        'year_str': year_str,
        'month_str': month_str,
        'day_str': day_str,
        'time_str': time_str,

    }

    if additional_dict:
        metadata_entry.update(additional_dict)

    # Use the full data path as the key
    metadata_dict[json_path] = metadata_entry

    # Save updated metadata
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata_dict, f)
    pass


def get_job_data(job_id, provider, additional_dict=None, checker_subprocess=False):
    """
    Fetches and returns job data and saves it in a structured directory.

    Parameters:
    - job_id (str): The ID of the job for which data should be fetched.
    - provider (obj): The provider object responsible for job execution.
    - additional_dict (dict): Additional metadata to be saved.

    Returns:
    memory (list/dict): The memory or counts dict of the job.

    Side effects:
    - Creates or navigates through existing directory structure and saves job data.
    """
    # Get global variables
    caller_frame = inspect.currentframe().f_back
    global_vars = caller_frame.f_globals

    # Get the notebookname
    notebook_name = os.path.basename(global_vars.get(
        '__vsc_ipynb_file__', 'not_defined_')).replace('.ipynb', '').replace(' ', '_')

    # Get the current date and time
    now = datetime.now()
    year_str = now.strftime('%Y')
    month_str = now.strftime('%m-%b')
    day_str = now.strftime('%d.%m.%y')
    time_str = now.strftime('%H:%M')

    # Create directory paths
    root_dir = find_and_create_scratch()
    if root_dir is None:
        print("Failed to set root_dir, exiting.")
        return None

    year_dir = f"{root_dir}/{year_str}"
    month_dir = f"{year_dir}/{month_str}"
    day_dir = f"{month_dir}/{day_str}"

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if job_id in filename:
                print(
                    f"A file with the job_id '{job_id}' already exists in: \n\n'{os.path.relpath(dirpath, start=root_dir)}/{filename}' \n\nLoading the corresponding data...")
                memory = load_memory_from_json(job_id)
                return memory

    # Create month and day directories if they do not exist
    for dir_path in [root_dir, year_dir, month_dir, day_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Retrieve the job
    job = provider.retrieve_job(job_id)

    # Check job status
    if job.status() in [JobStatus.RUNNING, JobStatus.QUEUED]:
        print(f"Job is {job.status()}. Waiting for job to finish...")

        if checker_subprocess:
            additional_dict_str = json.dumps(additional_dict)
            spawn_subprocess(job_id, provider, additional_dict_str, root_dir)

        return None
    
    elif job.status() == JobStatus.DONE:
        # Get the job result
        try:
            memory = job.result().get_memory()
            print("Job result has memory, getting memory...")
            memory_type = 'memory'
        except:
            memory = job.result().get_counts()
            print("Job result does not have memory, getting counts instead...")
            memory_type = 'counts'

        # Convert memory in the right format
        _is_complex = False
        if hasattr(memory, "tolist"):
            memory = memory.tolist()
            _is_complex = isinstance(memory[0][0], complex)
            print(f"Memory is complex/IQ Data: {_is_complex}")
            memory = make_serializable(memory)

        device_name = job.backend().name
        memory_shape = np.shape(memory)
        memory_shape_sci = tuple(format(dim, ".0e").replace(
            '+', '') if dim > 9000 else dim for dim in memory_shape)

        time_taken = job.result().time_taken

        # Prepare data and metadata for JSON file
        data_to_save = {
            "notebook_name": notebook_name,
            "device_name": device_name,
            "memory_shape": memory_shape,
            "memory": memory,
            "time_taken": time_taken

        }

        # Create the JSON file name
        json_filename = f"{time_str}-{notebook_name}-{device_name.replace('ibmq_', '').replace('ibm_', '')[:3]}-{memory_shape_sci}-{job_id}.json"
        json_filename = create_unique_filename(json_filename, day_dir)

        json_path = f"{day_dir}/{json_filename}"

        # Save to JSON file
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f)

        print(f"Data saved to {os.path.relpath(json_path, start=root_dir)}")

        # Update metadata file
        update_metadata(root_dir, year_str, month_str, day_str, time_str, json_path, job_id,
                        notebook_name, memory_shape, memory_type, _is_complex, time_taken, additional_dict)

        return memory

    else:
        print(f"Job is in an unrecognized state: {job.status()}. Exiting.")


def to_complex(obj):
    """
    Recursively convert lists of dictionaries to lists of complex numbers.
    """
    if isinstance(obj, list):
        return [to_complex(x) for x in obj]
    elif isinstance(obj, dict) and 'real' in obj and 'imag' in obj:
        return complex(obj['real'], obj['imag'])
    return obj


def load_memory_from_json(input=-1):
    """
    Load memory data from a JSON file given a specific date and time.

    Parameters:
    - input (str/int): String used to identify the correct JSON file, or, 
    if an integer is given, the index of the file in the list of matching files.
    Defaults to the most recent file.

    Returns:
    - memory: Loaded memory data if a uniquely matching file is found.
    - list: If multiple matching files are found.
    - None: If no matching file is found.
    """
    root_dir = find_and_create_scratch()

    if isinstance(input, str):
        print("Looking for a file matching the string: ", input)
        included_str = input
    elif isinstance(input, int):
        print("Looking for the memory with index: ", input)
        included_str = ''

    # Search for matching files
    search_pattern = f"{root_dir}/**/*{included_str}*.json"
    matching_files = glob.glob(search_pattern, recursive=True)

    # Check if a unique file is found
    if len(matching_files) == 1:
        with open(matching_files[0], 'r') as f:
            data = json.load(f)
            memory = data.get("memory", None)
            if memory is not None and isinstance(memory, list):
                memory = to_complex(memory)
            print(
                f"Data loaded from {os.path.relpath(matching_files[0], start=root_dir)}.")
            return memory
    elif len(matching_files) > 1:
        if isinstance(input, int):
            matching_files.sort(reverse=False)
            with open(matching_files[input], 'r') as f:
                data = json.load(f)
                memory = data.get("memory", None)
                if memory is not None and isinstance(memory, list):
                    memory = to_complex(memory)
                print(
                    f"Data loaded from {os.path.relpath(matching_files[input], start=root_dir)}.")
            return memory
        elif isinstance(input, str):
            print(
                f"Multiple files match the string: '{included_str}'. Please refine your query.")
            for filepath in matching_files:
                rel_path = os.path.relpath(filepath, start=root_dir)
                print(f"  - {rel_path}")
    else:
        print(f"No files match the string: '{included_str}'.")

    return None
