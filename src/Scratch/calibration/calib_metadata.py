# Maurice Hanisch mhanisc@ethz.ch
# Created 22.11.2023

import sys
sys.path.insert(0, r'/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/build')

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cpp.Probabilities import process_scaler_dict
import cpp_soft_info

from ..core import find_and_create_scratch
from .calibration_data import find_closest_calib_jobs

def generate_kde_grid(kde_dict, num_points, num_std_dev=3):
    """
    Generates a grid for Kernel Density Estimations (KDEs) using specified parameters.

    This function creates a grid over a specified range and evaluates the KDE for each qubit.
    It creates a grid for both the '0' and '1' states and stores the grid data in a dictionary.

    Args:
        kde_dict (dict): Dictionary containing KDE objects for each qubit.
        num_points (int): Number of points in each dimension of the grid.
        num_std_dev (int, optional): Number of standard deviations to define the grid range. Defaults to 3.

    Returns:
        dict: A dictionary where each key is a qubit index and each value is a tuple containing the grid's x and y coordinates, 
              and the densities for the '0' and '1' states.
    """
    grid_dict = {}

    # Define the grid range and create grid points
    grid_range_real = np.linspace(-num_std_dev, num_std_dev, num_points)
    grid_range_imag = np.linspace(-num_std_dev, num_std_dev, num_points)
    grid_x, grid_y = np.meshgrid(grid_range_real, grid_range_imag)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    for qubit_idx, (kde_0, kde_1) in kde_dict.items():
        # Evaluate KDE on the grid for both states
        grid_density_0 = (kde_0.score_samples(grid_points)).reshape(grid_x.shape)
        grid_density_1 = (kde_1.score_samples(grid_points)).reshape(grid_x.shape)
        
        # Create an instance of GridData and store in dictionary
        # grid_dict[qubit_idx] = cpp_probabilities.GridData(grid_x, grid_y, grid_density_0, grid_density_1)
        grid_dict[qubit_idx] = (grid_x, grid_y, grid_density_0, grid_density_1)

    return grid_dict


def update_grid_metadata(metadata_path, creation_date, backend_name, job_ids, grid_file_path, num_grid_points, num_std_dev):
    """
    Update the metadata file with new grid data or create an entry if it doesn't exist.

    This function adds or updates the metadata for a specific calibration grid. 
    It organizes the metadata by backend names and includes information about the 
    calibration job IDs, grid file paths, number of grid points, and standard deviations.

    Args:
        metadata_path (str): Path to the metadata file.
        creation_date (datetime or str): The creation date of the calibration jobs. 
                                         If a string, it should be in the format "yy-mm-dd_HhhM".
        backend_name (str): Name of the quantum computing backend.
        job_ids (dict): Dictionary of job IDs for each sampled state.
        grid_file_path (str): Path to the saved grid file.
        num_grid_points (int): Number of points in each dimension of the grid.
        num_std_dev (int): Number of standard deviations to cover in the grid.

    Returns:
        None: The function updates the metadata file but does not return anything.
    """
    # Convert creation_date to datetime if it is a string
    if isinstance(creation_date, str):
        creation_date = datetime.strptime(creation_date, "%y-%m-%d_%Hhh%M")

    # Check if the metadata file exists
    if os.path.exists(metadata_path):
        # Read existing metadata
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
    else:
        # Initialize metadata if file does not exist
        metadata = {}

    # Format the creation date
    formatted_date = creation_date.strftime("%y.%m.%d_%Hh%M")

    # Create a nested entry for the backend, if it doesn't exist
    if backend_name not in metadata:
        metadata[backend_name] = {}

    # Update or create the metadata entry for the specific backend and date
    formatted_key = f"{creation_date.strftime('%y.%m.%d_%Hh%M')}_{num_grid_points}pts_{num_std_dev}std"
    metadata[backend_name][formatted_key] = {
        'grid_file_path': grid_file_path,
        'job_ids': job_ids,
    }

    # Write the updated metadata back to the file
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, indent=4)


def save_grid(grid_dict, processed_scaler_dict, filename):
    """
    Save a grid dictionary and scaler dictionary as a JSON file.

    Args:
        grid_dict (dict): Dictionary containing tuples of NumPy arrays.
        scaler_dict (dict): Dictionary containing scaler parameters.
        filename (str): The name of the file to save the grid and scaler data.
    """
    serializable_grid_dict = {}
    for qubit_idx, (grid_x, grid_y, grid_density_0, grid_density_1) in grid_dict.items():
        serializable_grid_dict[qubit_idx] = {
            'grid_x': grid_x.tolist(), 
            'grid_y': grid_y.tolist(), 
            'grid_density_0': grid_density_0.tolist(), 
            'grid_density_1': grid_density_1.tolist()
        }

    data_to_save = {
        'grid_data': serializable_grid_dict,
        'scaler_data': processed_scaler_dict
    }

    with open(filename, 'w') as file:
        json.dump(data_to_save, file, indent=4)
    

def load_grid(filename):
    """
    Load a grid dictionary and scaler dictionary from a JSON file and create GridData objects.

    Args:
        filename (str): The name of the file to load the grid and scaler data from.

    Returns:
        tuple: A tuple containing two dictionaries, one with GridData objects and the other with scaler data.
    """
    with open(filename, 'r') as file:
        data_loaded = json.load(file)

    grid_dict = {}
    for qubit_idx, grid_data in data_loaded['grid_data'].items():
        grid_x = np.array(grid_data['grid_x'], dtype=np.double)
        grid_y = np.array(grid_data['grid_y'], dtype=np.double)
        grid_density_0 = np.array(grid_data['grid_density_0'], dtype=np.double)
        grid_density_1 = np.array(grid_data['grid_density_1'], dtype=np.double)

        # Create GridData objects
        grid_dict[int(qubit_idx)] = cpp_soft_info.GridData(
            grid_x, grid_y, grid_density_0, grid_density_1
        )

    scaler_dict = {}
    for qubit_idx, scaler_data in data_loaded['scaler_data'].items():
        mean_real, std_real = scaler_data[0]
        mean_imag, std_imag = scaler_data[1]
        scaler_dict[int(qubit_idx)] = ((float(mean_real), float(std_real)), (float(mean_imag), float(std_imag)))

    return grid_dict, scaler_dict



def create_or_load_kde_grid(provider, tobecalib_job: Optional[str] = None, tobecalib_backend: Optional[str] = None, 
                            num_grid_points: int = 300, num_std_dev: int = 3, other_date = None):
    """
    Create or load a Kernel Density Estimation (KDE) grid for a specified calibration job.
    If the grid already exists in the metadata, it loads the grid and scaler dictionaries. 
    If not, it generates a new grid, saves it, and updates the metadata.

    Args:
        provider: Quantum provider for obtaining calibration data.
        tobecalib_job (str): The job ID to be calibrated.
        num_grid_points (int): Number of points in each dimension of the grid.
        num_std_dev (int): Number of standard deviations for the grid range.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains GridData objects for each qubit.
               The second dictionary contains the processed scaler data for each qubit.
    """
    root_dir = find_and_create_scratch()
    grid_folder = root_dir + '/calibration_grids'
    metadata_file = root_dir + '/calibration_grid_metadata.json'

    # Check if metadata file exists and load it
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as file:
            grid_metadata = json.load(file)
    else:
        grid_metadata = {}

    # Find the closest calibration jobs
    closest_job_ids, backend, creation_date = find_closest_calib_jobs(tobecalib_job, tobecalib_backend, other_date=other_date)
    # print(f"Found jobs for backend {backend} with closest execution date {creation_date}. Retrieving kde grid...")

    # Format the creation date and construct the key for metadata
    formatted_key = f"{creation_date.strftime('%y.%m.%d_%Hh%M')}_{num_grid_points}pts_{num_std_dev}std"
    grid_entry = grid_metadata.get(backend, {}).get(formatted_key)
    print(f"Searching for {backend} and {formatted_key}")

    
    if grid_entry:
        # Load the existing grid and scaler
        grid_file_path = grid_entry['grid_file_path']
        loaded_grid_dict, loaded_scaler_dict = load_grid(grid_file_path)
        return loaded_grid_dict, loaded_scaler_dict
    else:
        from soft_info import get_KDEs # Lazy import to avoid circular imports
        # Retrieve KDEs and scaler data
        kde_dict, scaler_dict = get_KDEs(provider, tobecalib_job=tobecalib_job, 
                                         tobecalib_backend=tobecalib_backend, other_date=other_date)
        processed_scaler_dict = process_scaler_dict(scaler_dict)

        # Generate a new grid and save it
        grid_dict = generate_kde_grid(kde_dict, num_grid_points, num_std_dev)
        file_name = f"{formatted_key}_{str(closest_job_ids)}_grid.json"
        grid_file_path = os.path.join(grid_folder, file_name)
        save_grid(grid_dict, processed_scaler_dict, grid_file_path)

        # Update metadata
        update_grid_metadata(metadata_file, creation_date=creation_date, backend_name=backend, job_ids=closest_job_ids, grid_file_path=grid_file_path, num_grid_points=num_grid_points, num_std_dev=num_std_dev)
        
        # TODO: Change this to not load (Hacky fix because of some bug when cpp with created scaler dict)
        loaded_grid_dict, loaded_scaler_dict = load_grid(grid_file_path)
        return loaded_grid_dict, loaded_scaler_dict