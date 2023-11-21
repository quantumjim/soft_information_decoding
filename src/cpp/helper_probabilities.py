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
