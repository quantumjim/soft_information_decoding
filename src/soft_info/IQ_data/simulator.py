# Maurice D. Hanisch 
# Created 18.11.2023


import warnings

import numpy as np
from tqdm import tqdm
from qiskit_qec.circuits import RepetitionCodeCircuit
from qiskit_qec.noise import PauliNoiseModel
from qiskit_qec.utils import get_counts_via_stim

from ..Hardware.transpile_rep_code import get_repcode_layout, get_repcode_IQ_map
from ..Probabilities.KDE import get_KDEs

from qiskit_aer import AerSimulator
from qiskit_qec.utils.stim_tools import noisify_circuit

from Scratch import create_or_load_kde_grid


class RepCodeIQSimulator():
    def __init__(self, provider, distance: int, rounds: int, device: int, _is_hex: bool = True,
                 _resets: bool = False, other_date = None) -> None:
        self.provider = provider
        self.distance = distance
        self.rounds = rounds
        self.device = device
        self.other_date = other_date
        self.backend = self.provider.get_backend(self.device)
        self.layout = get_repcode_layout(self.distance, self.backend, _is_hex=_is_hex)
        self.qubit_mapping = get_repcode_IQ_map(self.layout, self.rounds)
        self.kde_dict, self.scaler_dict = get_KDEs(self.provider, tobecalib_backend=self.device, other_date=self.other_date)
        self.code = RepetitionCodeCircuit(self.distance, self.rounds, resets=_resets)
        self.grid_dict, self.processed_scaler_dict = create_or_load_kde_grid(self.provider, tobecalib_backend=self.device,
                                                           num_grid_points=300, num_std_dev=2, other_date=self.other_date)


    def get_noise_model(self, p1Q, p2Q, pXY, pZ, pRO, pRE) -> PauliNoiseModel:
        error_dict = {'reset': {"chan": {'i':1-pRE, 'x':pRE}},
                    'measure': {"chan": {'i':1-pRO, 'x':pRO}},
                    'h': {"chan": {'i':1-p1Q} | {i:p1Q/3 for i in 'xyz'}},
                    'idle_1': {"chan": {'i':1-pXY, 'x':pXY/2, 'y':pXY/2}},
                    'idle_2': {"chan": {'i':1-pZ, 'z':pZ}},
                    # 'cx': {"chan": {'ii':1-p2Q} | {i+j:p2Q/15 for i in 'ixyz' for j in 'ixyz' if i+j!='ii'}},
                    'cx': {"chan": {'ii':1-p2Q} | {'i'+i:p2Q/3 for i in 'xyz' }},                    
                    'swap': {"chan": {'ii':1-p2Q} | {i+j:p2Q/15 for i in 'ixyz' for j in 'ixyz' if i+j!='ii'}}}
        return PauliNoiseModel(fromdict=error_dict)
    
    def get_counts(self, shots: int, noise_model: PauliNoiseModel, logical: int) -> dict:
        warnings.warn("Getting counts via stim. This may take time...")
        # qc = self.code.circuit[str(logical)]
        # qc = noisify_circuit(qc, noise_model)
        # counts = AerSimulator().run(qc, shots=shots).result().get_counts()
        return get_counts_via_stim(self.code.circuit[str(logical)], shots=shots, noise_model=noise_model)
        # return counts
    
    def counts_to_IQ(self, counts: dict):
        total_shots = sum(counts.values())
        len_IQ_array = len(self.qubit_mapping)
        IQ_memory = np.zeros((total_shots, len_IQ_array), dtype=np.complex128)
        kde_samples_needed = {qubit_idx: {'0': 0, '1': 0} for qubit_idx in self.kde_dict.keys()}
        sample_counters = {qubit_idx: {'0': 0, '1': 0} for qubit_idx in self.kde_dict.keys()}    

        for count_str, shots in (counts.items()):
            num_spaces = 0
            inverted_count_str = count_str[::-1]
            for IQ_idx, bit in enumerate(inverted_count_str):
                if bit == ' ':
                    num_spaces += 1
                    continue
                qubit_idx = self.qubit_mapping[IQ_idx - num_spaces]
                kde_samples_needed[qubit_idx][bit] += shots

        kde_samples = {}
        for qubit_idx, needed_nb_samples in kde_samples_needed.items():
            [kde0, kde1], scaler = self.kde_dict[qubit_idx], self.scaler_dict[qubit_idx]
            if needed_nb_samples['0'] > 0:
                samples0 = scaler.inverse_transform(kde0.sample(needed_nb_samples['0'], random_state=42))
            else:
                samples0 = np.empty((0, 2)) 
            if needed_nb_samples['1'] > 0:
                samples1 = scaler.inverse_transform(kde1.sample(needed_nb_samples['1'], random_state=42))
            else:
                samples1 = np.empty((0, 2)) 
            kde_samples[qubit_idx] = {'0': samples0, '1': samples1}
        
        shot_idx = 0
        for count_str, shots in tqdm(counts.items()):
            for _ in range(shots):
                num_spaces = 0
                inverted_count_str = count_str[::-1]
                for IQ_idx, bit in enumerate(inverted_count_str):
                    if bit == ' ':
                        num_spaces += 1
                        continue
                    cIQ_idx = IQ_idx - num_spaces
                    qubit_idx = self.qubit_mapping[cIQ_idx]
                    sample_index = sample_counters[qubit_idx][bit]
                    sample = kde_samples[qubit_idx][bit][sample_index]   
                    IQ_memory[shot_idx, cIQ_idx] = complex(sample[0], sample[1])
                    sample_counters[qubit_idx][bit] += 1
                shot_idx += 1

        assert sample_counters == kde_samples_needed
    
        return IQ_memory
    

    def generate_IQ_dict(self) -> dict:        
        IQ_dict = {}
        for qubit_idx, grid in self.grid_dict.items():
            # Calculate the differences
            grid_diff_max = grid.grid_density_0 - grid.grid_density_1
            grid_diff_min = grid.grid_density_1 - grid.grid_density_0

            # Apply coordinate restriction
            mask = (grid.grid_x > -1) & (grid.grid_x < 1) & (grid.grid_y > -1) & (grid.grid_y < 1)
            restricted_diff_max = np.where(mask, grid_diff_max, -np.inf)
            restricted_diff_min = np.where(mask, np.where(grid.grid_density_1 > grid.grid_density_0, grid_diff_min, np.inf), np.inf)

            # Get the coordinates of the maximum and minimum differences
            max_diff_coordinate = np.unravel_index(np.argmax(restricted_diff_max), grid_diff_max.shape)
            min_diff_coordinate = np.unravel_index(np.argmin(restricted_diff_min), grid_diff_min.shape)

            # Get the (x, y) coordinates for both
            x_max, y_max = grid.grid_x[max_diff_coordinate[0], max_diff_coordinate[1]], grid.grid_y[max_diff_coordinate[0], max_diff_coordinate[1]]
            x_min, y_min = grid.grid_x[min_diff_coordinate[0], min_diff_coordinate[1]], grid.grid_y[min_diff_coordinate[0], min_diff_coordinate[1]]

            # Get the densities at max and min coordinates
            density_0_safe = grid.grid_density_0[max_diff_coordinate[0], max_diff_coordinate[1]]
            density_1_safe = grid.grid_density_1[max_diff_coordinate[0], max_diff_coordinate[1]]
            density_0_at_min = grid.grid_density_0[min_diff_coordinate[0], min_diff_coordinate[1]]
            density_1_at_min = grid.grid_density_1[min_diff_coordinate[0], min_diff_coordinate[1]]

            # Get the scaler parameters
            (mean_real, std_real), (mean_imag, std_imag) = self.processed_scaler_dict[qubit_idx]

            # Inverse transform the coordinates
            x_max_rescaled = x_max * std_real + mean_real
            y_max_rescaled = y_max * std_imag + mean_imag
            x_min_rescaled = x_min * std_real + mean_real
            y_min_rescaled = y_min * std_imag + mean_imag

            # Create complex IQ points
            iq_point_max = complex(x_max_rescaled, y_max_rescaled)
            iq_point_min = complex(x_min_rescaled, y_min_rescaled)

            # Append or create new entry in result_dict
            if qubit_idx not in IQ_dict:
                IQ_dict[qubit_idx] = {
                "max_diff_0_coordinate": (x_max, y_max),
                "max_diff_0_coordinate_rescaled": (x_max_rescaled, y_max_rescaled),
                "min_diff_1_coordinate": (x_min, y_min),
                "min_diff_1_coordinate_rescaled": (x_min_rescaled, y_min_rescaled),
                "iq_point_safe": iq_point_max,
                "iq_point_ambig": iq_point_min,
                "density_0_safe": density_0_safe,
                "density_1_safe": density_1_safe,
                "density_0_ambig": density_0_at_min,
                "density_1_ambig": density_1_at_min
            }

        return IQ_dict   
    

    def counts_to_IQ_extreme(self, p_ambig: float in (0, 1), 
                            IQ_dict: dict, counts: dict):
        total_shots = sum(counts.values())
        IQ_memory = np.zeros((total_shots, len(self.qubit_mapping)), dtype=np.complex128)

        shot_idx = 0
        for count_str, shots in tqdm(counts.items()):
            for _ in (range(shots)):
                num_spaces = 0
                inverted_count_str = count_str[::-1]
                for IQ_idx, bit in enumerate(inverted_count_str):
                    if bit == ' ':
                        num_spaces += 1
                        continue
                    cIQ_idx = IQ_idx - num_spaces
                    qubit_idx = self.qubit_mapping[cIQ_idx]
                    
                    # sample with p_ambig
                    _ambig = np.random.choice([False, True], p=[1-p_ambig, p_ambig])
                    point_str = "safe" if not _ambig else "ambig"
                    if num_spaces == self.rounds:
                        point_str = "safe"
                    sample = IQ_dict[qubit_idx]["iq_point_"+point_str]

                    IQ_memory[shot_idx, cIQ_idx] = sample
                shot_idx += 1
                
        return IQ_memory


    def generate_IQ(self, shots: int, noise_model: PauliNoiseModel, logical: int) -> list:
        counts = self.get_counts(shots, noise_model, logical)
        IQ_memory = self.counts_to_IQ(counts)
        return IQ_memory
    
    
    def generate_extreme_IQ(self, shots: int, p_ambig: float in (0, 1), 
                            noise_model: PauliNoiseModel = None) -> list:
        IQ_dict = self.generate_IQ_dict()
        counts = self.get_counts(shots, noise_model, logical=0) # hardcoded for logical 0
        IQ_memory = self.counts_to_IQ_extreme(p_ambig, IQ_dict, counts)
        return IQ_memory




        
        
