# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-20

import numpy as np


def process_scaler_dict(scaler_dict):
    processed_dict = {}
    for qubit_idx, scaler in scaler_dict.items():
        # Assuming the scaler is fit on complex data with real and imaginary parts as separate features
        # Hence, the mean_ and scale_ arrays should have two elements each
        if len(scaler.mean_) != 2 or len(scaler.scale_) != 2:
            raise ValueError(f"Scaler for qubit {qubit_idx} is not fit on complex data.")
        
        mean_real, mean_imag = scaler.mean_
        std_real, std_imag = scaler.scale_
        processed_dict[qubit_idx] = ((mean_real, std_real), (mean_imag, std_imag))

    return processed_dict


# def generate_grid_dict(scaler_dict, kde_dict, )
# grid_dict = {}
# num_points = 10
# for qubit_idx, (kde_0, kde_1) in kde_dict.items():
#     scaler = scaler_dict[qubit_idx]

#     # Retrieve the dataset for this qubit and split into real and imaginary parts
#     data = np.array(memory[0])  # TODO change that: Retrieving data from memory 0 to get the min and max values
#     data_real_imag = np.column_stack([np.real(data), np.imag(data)])

#     # Scale data
#     scaled_data = scaler.transform(data_real_imag)

#     # Create grid
#     grid_x, grid_y = np.linspace(scaled_data[:, 0].min() - 1, scaled_data[:, 0].max() + 1, num_points), \
#                      np.linspace(scaled_data[:, 1].min() - 1, scaled_data[:, 1].max() + 1, num_points)
#     grid_x, grid_y = np.meshgrid(grid_x, grid_y)
#     grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

#     # Evaluate KDE on the grid for both states
#     grid_density_0 = (kde_0.score_samples(grid_points)).reshape(grid_x.shape)
#     grid_density_1 = (kde_1.score_samples(grid_points)).reshape(grid_x.shape)
    
#     # Create an instance of GridData and store in dictionary
#     grid_dict[qubit_idx] = cpp_probabilities.GridData(grid_x, grid_y, grid_density_0, grid_density_1)