import sys
sys.path.insert(0, r'/Users/mha/My Drive/Desktop/Studium/Physik/MSc/Semester 3/IBM/IBM GIT/Soft-Info/build')

import random

import numpy as np
import pytest

import cpp_probabilities


def create_your_grid_data(num_points):
    # Replace this with your actual grid data creation logic
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, num_points), np.linspace(0, 1, num_points))
    grid_density_0 = np.random.rand(num_points, num_points)  # Placeholder values
    grid_density_1 = np.random.rand(num_points, num_points)  # Placeholder values
    return grid_x, grid_y, grid_density_0, grid_density_1

def create_specific_grid_data(num_points):
    # Create a meshgrid
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, num_points), np.linspace(0, 1, num_points))

    # Create grid densities: all zeros for grid_density_0 and all 0.5s for grid_density_1
    grid_density_0 = np.zeros((num_points, num_points))
    grid_density_1 = np.full((num_points, num_points), 0.5)

    return grid_x, grid_y, grid_density_0, grid_density_1


def generate_random_qubit_mapping(num_keys):
    """
    Generates a random qubit mapping.

    Args:
    num_keys (int): The number of keys in the mapping.

    Returns:
    dict: A dictionary where keys are mapped randomly to either 0 or 1.
    """
    qubit_mapping = {}
    for i in range(num_keys):
        qubit_mapping[i] = random.randint(0, 1)
    return qubit_mapping


def test_get_counts_assertion_synd_rounds():

    kde_grid_dict = {}
    for qubit_idx in range(2):  # Assuming two qubits
        grid_x, grid_y, grid_density_0, grid_density_1 = create_your_grid_data(100)
        kde_grid_dict[qubit_idx] = cpp_probabilities.GridData(grid_x, grid_y, grid_density_0, grid_density_1)
    
    len_IQ = 9   # Number of keys in the mapping
    qubit_mapping = generate_random_qubit_mapping(len_IQ)

    scaled_IQ_data_np = np.random.rand(int(1e1), len_IQ*2)
    scaled_IQ_data = cpp_probabilities.numpy_to_eigen(scaled_IQ_data_np)

    synd_rounds_list = [-1, 2, 4, 5, 6, 19]
    for synd_rounds in synd_rounds_list:
        with pytest.raises(RuntimeError):
            counts = cpp_probabilities.get_counts(scaled_IQ_data, qubit_mapping, kde_grid_dict, synd_rounds)


def test_get_counts_all_same():
    # Create specific grid data
    kde_grid_dict = {}
    kde_grid_dict2 = {}
    for qubit_idx in range(2):  # Assuming two qubits
        grid_x, grid_y, grid_density_0, grid_density_1 = create_specific_grid_data(10)
        kde_grid_dict[qubit_idx] = cpp_probabilities.GridData(grid_x, grid_y, grid_density_0, grid_density_1)
        kde_grid_dict2[qubit_idx] = cpp_probabilities.GridData(grid_x, grid_y, grid_density_1, grid_density_0)
    
    # Create a random qubit mapping and scaled IQ data
    qubit_mapping = {i: i % 2 for i in range(20)}  # Example mapping
    scaled_IQ_data_np = np.random.rand(10, 20)  # 10 shots, 20 columns (10 measurements)
    scaled_IQ_data = cpp_probabilities.numpy_to_eigen(scaled_IQ_data_np)

    synd_rounds = 0

    # Call the get_counts function
    counts = cpp_probabilities.get_counts(scaled_IQ_data, qubit_mapping, kde_grid_dict, synd_rounds)
    counts2 = cpp_probabilities.get_counts(scaled_IQ_data, qubit_mapping, kde_grid_dict2, synd_rounds)

    # Check if all outcomes in counts are '1'
    for outcome in counts:
        assert all(char == '1' for char in outcome), "Not all outcomes are 1s"
    for outcome in counts2:
        assert all(char == '0' for char in outcome), "Not all outcomes are 0s"


def test_get_counts_spacing():
    # Create specific grid data
    kde_grid_dict = {}
    kde_grid_dict2 = {}
    for qubit_idx in range(2):  # Assuming two qubits
        grid_x, grid_y, grid_density_0, grid_density_1 = create_specific_grid_data(10)
        kde_grid_dict[qubit_idx] = cpp_probabilities.GridData(grid_x, grid_y, grid_density_0, grid_density_1)
        kde_grid_dict2[qubit_idx] = cpp_probabilities.GridData(grid_x, grid_y, grid_density_1, grid_density_0)
    
    # Create a random qubit mapping and scaled IQ data
    qubit_mapping = {i: i % 2 for i in range(9)}  
    scaled_IQ_data_np = np.random.rand(10, 18)  
    scaled_IQ_data = cpp_probabilities.numpy_to_eigen(scaled_IQ_data_np)

    synd_rounds = 3

    # Call the get_counts function
    counts = cpp_probabilities.get_counts(scaled_IQ_data, qubit_mapping, kde_grid_dict, synd_rounds)
    
    for outcome in counts:
        sections = outcome.split(' ')
        assert len(sections) == synd_rounds+1, "Outcome in counts does not have T+1 = 4 sections: '--- -- -- --'"
        for idx, section in enumerate(sections):
            if idx == 0:
                assert len(section) == 3, "Code section of d=3, T=3 rep code is not length 3"
            else:
                assert len(section) == 2, "Synd Section of d=3, T=3 rep code is not length 2"



    

