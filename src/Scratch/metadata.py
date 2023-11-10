# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-20

import os 
import json

import pandas as pd

from .core import find_and_create_scratch, get_notebook_name


def metadata_helper(*args, **kwargs):
    """
    A helper function to collect and organize metadata.

    Parameters:
    n_shots (Union[int, float]): The number of shots for the experiment.
    meas_level (Union[1, 2]): The measurement level, either 1 or 2.
    *args: Additional arguments.
    **kwargs: Additional keyword arguments.

    Returns:
    Dict[str, Any]: A dictionary containing the collected metadata.
    """

    # Organize the metadata into a dictionary
    metadata = {

    }

    # If there are additional arguments or keyword arguments, include them in the metadata
    if args:
        metadata['additional_args'] = args
    if kwargs:
        metadata.update(kwargs)

    return metadata


def metadata_loader(_extract: bool = False, _drop_inutile : bool = False):
    """
    Loads metadata related to job submissions in order of creation date.

    Returns:
        pandas.DataFrame: DataFrame containing metadata for each job.

    1) Drop rows with NaN values:
        >>> cleaned_data = metadata.dropna(subset=['column_name'])

    2) Filter rows where 'backend_name' is 'ibmq_jakarta':
        >>> filtered_data = metadata[metadata['backend_name'] == 'ibmq_jakarta']

    3) Expand df with columns from 'backend_options':
        >>> expanded_df = metadata.join(filtered_data['backend_options'].apply(pd.Series))

    4) Filter rows where 'backend_options' has 'shots' equal to 1e6:
        >>> filtered_data = metadata[metadata['backend_options'].apply(lambda x: x.get('shots')) == 1e6]

    5) Drop specific columns:
        >>> reduced_data = metadata.drop(columns=['column_name1', 'column_name2'])

    6) Get the job_id's of selected data:
        >>> job_ids = filtered_data['job_id']

    7) Sort data by a specific column:
       >>> sorted_data = metadata.sort_values(by='column_name', ascending=True)

    8) Filter rows where 'backend_name' is 'ibmq_jakarta' AND 'notebook_name' is 'notebook_name':
        >>> filtered_data = metadata[(metadata['backend_name'] == 'ibmq_jakarta') & (metadata['notebook_name'] == 'notebook_name')]

    9) Filter rows where 'backend_name' is 'ibmq_jakarta' OR 'notebook_name' is 'notebook_name':
        >>> filtered_data = metadata[(metadata['backend_name'] == 'ibmq_jakarta') | (metadata['notebook_name'] == 'notebook_name')]

    10) Find unique values in a column and their counts:
        >>> unique_counts = df['backend_name'].value_counts()   
    """
    root_dir = find_and_create_scratch()
    metadata_path = f"{root_dir}/job_metadata.json"
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("job_metadata.json file not found.")
    
    metadata = pd.read_json(metadata_path)
    if _extract:
        metadata = metadata.join(metadata.backend_options.apply(
            pd.Series), rsuffix='_xp').join(metadata.additional_metadata.apply(pd.Series))
        metadata = metadata.drop(
            columns=['backend_options', 'additional_metadata'])
    if _drop_inutile:
        metadata = metadata.drop(columns=["meas_level", "init_qubits", "job_metadata", "job_name", "meas_return", "skip_transpilation", "memory"])
    return metadata.sort_values(by='creation_date', ascending=False)


def update_metadata(job, backend_name, additional_dict: dict):
    '''Updates the metadata file with MANDATORY additional dicts.'''

    ROOT_DIR = find_and_create_scratch()
    notebook_name = get_notebook_name()

    metadata_file_path = os.path.join(ROOT_DIR, f"job_metadata.json")

    # Load or create the metadata dictionary
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, 'r') as f:
            metadata_list = json.load(f)
    else:
        metadata_list = []

    # Find the existing metadata entry by job_id
    for entry in metadata_list:
        if entry.get('job_id') == job.job_id():
            if entry['additional_metadata'] is None:
                entry['additional_metadata'] = {}
            entry['additional_metadata'].update(additional_dict)
            break
    else:
        # If not found, create new metadata entry
        metadata_entry = {
            "creation_date": str(job.creation_date()),
            'notebook_name': notebook_name,
            'backend_name': backend_name,
            'backend_options': job.backend_options(),
            'additional_metadata': additional_dict,
            'job_id': job.job_id(),
            'job_name': job.name(),
            'job_metadata': job.metadata,
            "tags": job.tags(),
        }
        metadata_list.append(metadata_entry)

    # Save updated metadata
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    pass
