# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-20

import json
import os
from datetime import datetime
import numpy as np
from qiskit.providers import JobStatus


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


def get_job_data(job_id, provider, global_vars=globals()):
    """
    Fetches and returns job data and saves it in a structured directory.
    
    Parameters:
    - job_id (str): The ID of the job for which data should be fetched.
    - provider (obj): The provider object responsible for job execution.
    - global_vars (dict, optional): The global variables from the calling script. 
      Defaults to None. To get the correct notebook name, 
      explicitly pass globals() when calling this function.
    
    Returns:
    memory (list/dict): The memory or counts dict of the job.
    
    Side effects:
    - Creates or navigates through existing directory structure and saves job data.
    
    """
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

    # Create month and day directories if they do not exist
    for dir_path in [root_dir, year_dir, month_dir, day_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Retrieve the job
    job = provider.retrieve_job(job_id)

    # Check job status
    if job.status() in [JobStatus.RUNNING, JobStatus.QUEUED]:
        print(f"Job is {job.status()}")
        return None
    elif job.status() == JobStatus.DONE:
        # Get the job result
        try:
            memory = job.result().get_memory()
            print("Job result has memory, getting memory")
        except:
            memory = job.result().get_counts()
            print("Job result does not have memory, getting counts instead")

        # Convert memory in the right format
        if hasattr(memory, "tolist"):
            memory = memory.tolist()
            memory = make_serializable(memory)

        device_name = job.backend().name

        # Prepare data and metadata for JSON file
        data_to_save = {
            "notebook_name": notebook_name,
            "device_name": device_name,
            "memory_shape": np.shape(memory),
            "memory": memory
        }

        # Create the JSON file name
        json_filename = f"{time_str}{notebook_name}-{device_name.replace('ibmq_', '').replace('ibm_', '')}.json"
        json_filename = create_unique_filename(json_filename, day_dir)

        json_path = f"{day_dir}/{json_filename}"

        # Save to JSON file
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f)

        print(f"Data saved to {json_path}")

        return memory

    else:
        print(f"Job is in an unrecognized state: {job.status()}")
