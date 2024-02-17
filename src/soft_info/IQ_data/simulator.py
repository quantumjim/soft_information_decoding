# Maurice D. Hanisch 
# Created 18.11.2023


import warnings
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from qiskit_qec.circuits import RepetitionCodeCircuit
from qiskit_qec.noise import PauliNoiseModel
from qiskit_qec.utils import get_counts_via_stim
import stim

from ..Hardware.transpile_rep_code import get_repcode_layout, get_repcode_IQ_map
from ..Probabilities.KDE import get_KDEs

from qiskit_aer import AerSimulator
from qiskit_qec.utils.stim_tools import noisify_circuit

from Scratch import create_or_load_kde_grid


class RepCodeIQSimulator():

    _kde_cache = OrderedDict()
    _grid_cache = OrderedDict()
    _max_cache_size = 2

    def __init__(self, provider, distance: int, rounds: int, device: int, _is_hex: bool = True,
                 _resets: bool = False, other_date = None) -> None:
        self.provider = provider
        self.distance = distance
        self.rounds = rounds
        self.device = device
        self.other_date = other_date
        self.backend = self.provider.get_backend(self.device)
        self.layout = get_repcode_layout(self.distance, self.backend, _is_hex=_is_hex) if self.device != 'ibm_torino' else RepCodeIQSimulator.get_layout_torino(self.distance) # Hardcoded torino layout
        self.qubit_mapping = get_repcode_IQ_map(self.layout, self.rounds)
        self._resets = _resets
        # self.code = RepetitionCodeCircuit(self.distance, self.rounds, resets=_resets)
        self.stim_circ = None

        kde_cache_key = (device, other_date)
        grid_cache_key = (device, other_date)

        if kde_cache_key in RepCodeIQSimulator._kde_cache:
            self.kde_dict, self.scaler_dict = RepCodeIQSimulator._kde_cache[kde_cache_key]
            RepCodeIQSimulator._kde_cache.move_to_end(kde_cache_key)
        else:
            self.kde_dict, self.scaler_dict = get_KDEs(self.provider, tobecalib_backend=self.device, other_date=self.other_date)
            RepCodeIQSimulator._update_cache(RepCodeIQSimulator._kde_cache, kde_cache_key, (self.kde_dict, self.scaler_dict))

        if grid_cache_key in RepCodeIQSimulator._grid_cache:
            self.grid_dict, self.processed_scaler_dict = RepCodeIQSimulator._grid_cache[grid_cache_key]
            RepCodeIQSimulator._grid_cache.move_to_end(grid_cache_key)
        else:
            self.grid_dict, self.processed_scaler_dict = create_or_load_kde_grid(self.provider, tobecalib_backend=self.device, num_grid_points=300, num_std_dev=2, other_date=self.other_date)
            RepCodeIQSimulator._update_cache(RepCodeIQSimulator._grid_cache, grid_cache_key, (self.grid_dict, self.processed_scaler_dict))


    # Hardcoded torino layout (UGLY!)
    @staticmethod
    def get_layout_torino(distance):
        import pickle
        with open('longest_path_torino.pkl', 'rb') as f:
            path = pickle.load(f)
        bounded_path = path[:2 * distance - 1]
        layout = bounded_path[1::2] + bounded_path[::2] 
    
        return layout

    @staticmethod
    def _update_cache(cache: OrderedDict, key, value):
        if len(cache) >= RepCodeIQSimulator._max_cache_size:
            cache.popitem(last=False)  # Remove the oldest item
        cache[key] = value

    @staticmethod
    def get_noise_model(p1Q, p2Q, pXY, pZ, pRO, pRE) -> PauliNoiseModel:
        error_dict = {'reset': {"chan": {'i':1-pRE, 'x':pRE}},
                    'measure': {"chan": {'i':1-pRO, 'x':pRO}},
                    'h': {"chan": {'i':1-p1Q} | {i:p1Q/3 for i in 'xyz'}},
                    'idle_1': {"chan": {'i':1-pXY, 'x':pXY/2, 'y':pXY/2}},
                    'idle_2': {"chan": {'i':1-pZ, 'z':pZ}},
                    # 'cx': {"chan": {'ii':1-p2Q} | {i+j:p2Q/15 for i in 'ixyz' for j in 'ixyz' if i+j!='ii'}},
                    'cx': {"chan": {'ii':1-p2Q} | {'i'+i:p2Q/3 for i in 'xyz' }},                    
                    'swap': {"chan": {'ii':1-p2Q} | {i+j:p2Q/15 for i in 'ixyz' for j in 'ixyz' if i+j!='ii'}}}
        return PauliNoiseModel(fromdict=error_dict)
    
    def get_stim_circuit(self, noise_list: list) -> stim.Circuit:
        """Returns a stim circuit with the given noise parameters.        
        Args:
            noise_list (list): List of noise parameters [two-qubit-fidelity, reset error, measurement error, idle error].
        """
        circuit = stim.Circuit.generated("repetition_code:memory",
                                distance=self.distance,
                                rounds=self.rounds,
                                after_clifford_depolarization=noise_list[0], #two-qubit-fidelity,
                                after_reset_flip_probability=noise_list[1], #reset error,
                                before_measure_flip_probability=noise_list[2], #measurement error,
                                before_round_data_depolarization=noise_list[3]) #idle error)
        self.stim_circ = circuit

    # def vectorized_processing(self, meas_outcomes):
    #     # Step 1: Convert to '0' and '1'
    #     meas_strings = np.where(meas_outcomes, '1', '0')

    #     # Step 2: Insert spaces and reverse. For vectorization, we work with array manipulation
    #     # Calculate the positions where spaces should be added (accounting for reversing)
    #     space_indices = np.arange(self.distance - 2, self.rounds * (self.distance - 1), self.distance - 1)

    #     # Initialize an empty list to store processed strings
    #     processed_strings = []

    #     # Process each measurement outcome
    #     for row in meas_strings:
    #         # Convert row to a list of characters for easier manipulation
    #         char_list = list(row)
            
    #         # Insert spaces at the calculated positions
    #         for index in space_indices:
    #             if index < len(char_list):
    #                 char_list.insert(index + 1, ' ')
            
    #         # Reverse the string and join
    #         processed_string = ''.join(char_list[::-1])
    #         processed_strings.append(processed_string)

    #     # Convert processed strings back to an array
    #     processed_array = np.array(processed_strings)

    #     # Step 3: Count occurrences of each unique string
    #     unique, counts = np.unique(processed_array, return_counts=True)
    #     counts_dict = dict(zip(unique, counts))

    #     return counts_dict
        
    def get_counts(self, shots: int, stim_circuit: stim.Circuit, verbose=False) -> dict:
        meas_outcomes = stim_circuit.compile_sampler(seed=42).sample(shots)
        print("generated counts")
        counts = {}
        for row in meas_outcomes:
            count_str = ''
            for nb, bit in enumerate(row):
                count_str += '0' if bit == False else '1'
                if (nb+1) % (self.distance-1) == 0 and nb < self.rounds*(self.distance-1):
                    count_str += ' ' 
            count_str = count_str[::-1]
            if count_str in counts:
                counts[count_str] += 1
            else:
                counts[count_str] = 1

        # counts = self.vectorized_processing(meas_outcomes)
        
        print("correcting counts")
        
        # correct samples to have no reset counts if _resets == False
        if self._resets == False:
            no_reset_counts = {}
            for count_key, shots in counts.items():
                parts=count_key.split(" ")
                print("parts:", parts) if verbose else None
                count_part = parts[0]
                print("count_part:", count_part) if verbose else None  
                check_parts = parts[1:]
                print("check_parts:", check_parts, "\n") if verbose else None

                for i in range(len(check_parts)):
                    if i == 0:
                        print("skipped last check part:", check_parts[-1], "\n") if verbose else None
                        continue
                    current_check_str = check_parts[-(i+1)]
                    print("current_check_str:", current_check_str) if verbose else None
                    prev_check_str = check_parts[-i]
                    print("prev_check_str:", prev_check_str) if verbose else None
                    new_check_str = ''
                    for bit1, bit2 in zip(prev_check_str, current_check_str):
                        new_check_str += str(int(bit1)^int(bit2))
                    print("new_check_str:", new_check_str, "\n") if verbose else None
                    check_parts[-(i+1)] = new_check_str

                print("\ncheck_parts after modulo:", check_parts) if verbose else None

                new_count_str = count_part + " " +  ' '.join(check_parts)
                print("\nnew_count_str:", new_count_str) if verbose else None

                
                if new_count_str in no_reset_counts:
                    no_reset_counts[new_count_str] += shots
                else:
                    no_reset_counts[new_count_str] = shots
            counts = no_reset_counts
        
        print("finished correcting counts")
        return counts



        
    # def get_counts(self, shots: int, noise_model: PauliNoiseModel, logical: int) -> dict:
    #     warnings.warn("Getting counts via stim. This may take time...")
    #     # qc = self.code.circuit[str(logical)]
    #     # qc = noisify_circuit(qc, noise_model)
    #     # counts = AerSimulator().run(qc, shots=shots).result().get_counts()
    #     return get_counts_via_stim(self.code.circuit[str(logical)], shots=shots, noise_model=noise_model)
    #     # return counts

    
    def counts_to_IQ(self, counts: dict, verbose = False):
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
        iterable = tqdm(counts.items()) if verbose else counts.items()
        for count_str, shots in iterable:
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
    
    def _rescale_coordinates(self, grid, coordinates, qubit_idx):
        """Helper function to rescale coordinates."""
        (mean_real, std_real), (mean_imag, std_imag) = self.processed_scaler_dict[qubit_idx]
        x_rescaled = grid.grid_x[coordinates[0], coordinates[1]] * std_real + mean_real
        y_rescaled = grid.grid_y[coordinates[0], coordinates[1]] * std_imag + mean_imag
        return x_rescaled, y_rescaled
        

    def generate_IQ_dict(self) -> dict:
        IQ_dict = {}
        for qubit_idx, grid in self.grid_dict.items():
            # print()
            # print("qubit_idx", qubit_idx)
            # Calculate the differences for both states
            grid_diff_0 = grid.grid_density_0 - grid.grid_density_1  # Max diff for state '0'
            grid_diff_1 = grid.grid_density_1 - grid.grid_density_0  # Max diff for state '1'

            # Apply coordinate restriction
            mask = (grid.grid_x > -1) & (grid.grid_x < 1) & (grid.grid_y > -1) & (grid.grid_y < 1)
            restricted_diff_max_0 = np.where(mask, grid_diff_0, -np.inf)
            restricted_diff_min_0 = np.where(mask, np.where(grid.grid_density_1 > grid.grid_density_0, grid_diff_1, np.inf), np.inf)
            restricted_diff_max_1 = np.where(mask, grid_diff_1, -np.inf)
            restricted_diff_min_1 = np.where(mask, np.where(grid.grid_density_0 > grid.grid_density_1, grid_diff_0, np.inf), np.inf)

            # Get the coordinates of the maximum and minimum differences for both states
            max_diff_coordinate_0 = np.unravel_index(np.argmax(restricted_diff_max_0), grid_diff_0.shape)
            min_diff_coordinate_0 = np.unravel_index(np.argmin(restricted_diff_min_0), grid_diff_0.shape)
            max_diff_coordinate_1 = np.unravel_index(np.argmax(restricted_diff_max_1), grid_diff_1.shape)
            min_diff_coordinate_1 = np.unravel_index(np.argmin(restricted_diff_min_1), grid_diff_1.shape)

            # print("density_diff at max_diff_coordinate_0", grid_diff_0[max_diff_coordinate_0])
            # print("density_diff at min_diff_coordinate_0", grid_diff_0[min_diff_coordinate_0])
            # print("density_diff_0 at max_diff_coordinate_1", grid_diff_0[max_diff_coordinate_1])
            # print("density_diff_0 at min_diff_coordinate_1", grid_diff_0[min_diff_coordinate_1])
 
            # Get the scaler parameters
            (mean_real, std_real), (mean_imag, std_imag) = self.processed_scaler_dict[qubit_idx]

            # Inverse transform the coordinates
            def rescale(x, y):
                return x * std_real + mean_real, y * std_imag + mean_imag

            # Create complex IQ points for both states
            iq_point_safe_0 = complex(*rescale(grid.grid_x[max_diff_coordinate_0], grid.grid_y[max_diff_coordinate_0]))
            # print("grid_point_safe_0", (grid.grid_x[max_diff_coordinate_0], grid.grid_y[max_diff_coordinate_0]))
            iq_point_ambig_0 = complex(*rescale(grid.grid_x[min_diff_coordinate_0], grid.grid_y[min_diff_coordinate_0]))
            # print("grid_point_ambig_0", (grid.grid_x[min_diff_coordinate_0], grid.grid_y[min_diff_coordinate_0]))
            iq_point_safe_1 = complex(*rescale(grid.grid_x[max_diff_coordinate_1], grid.grid_y[max_diff_coordinate_1]))
            # print("grid_point_safe_1", (grid.grid_x[max_diff_coordinate_1], grid.grid_y[max_diff_coordinate_1]))
            iq_point_ambig_1 = complex(*rescale(grid.grid_x[min_diff_coordinate_1], grid.grid_y[min_diff_coordinate_1]))
            # print("grid_point_ambig_1", (grid.grid_x[min_diff_coordinate_1], grid.grid_y[min_diff_coordinate_1]))

            # Append to IQ_dict
            IQ_dict[qubit_idx] = {
                "iq_point_safe_0": iq_point_safe_0,
                "iq_point_ambig_0": iq_point_ambig_0,
                "iq_point_safe_1": iq_point_safe_1,
                "iq_point_ambig_1": iq_point_ambig_1
            }

        return IQ_dict

    

    def counts_to_IQ_extreme(self, p_ambig: float, 
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
                    sample = IQ_dict[qubit_idx]["iq_point_"+point_str+"_"+bit]

                    IQ_memory[shot_idx, cIQ_idx] = sample
                shot_idx += 1
                
        return IQ_memory


    def generate_IQ(self, shots: int, noise_list: list, logical: int = None, verbose = False) -> list:
        """Generates IQ data for the given logical state.
        Args:
            shots (int): Number of shots.
            noise_list (list): List of noise parameters [two-qubit-fidelity, reset error, measurement error, idle error].
            logical (int, optional): Logical state. Defaults to None.
        """
        warnings.warn("Logical != 0 is currently not supported") if logical is not None else None
        noise_list = [1e-8, 1e-8, 1e-8, 1e-8] if noise_list is None else noise_list
        self.get_stim_circuit(noise_list)
        counts = self.get_counts(shots, self.stim_circ, verbose)
        IQ_memory = self.counts_to_IQ(counts, verbose)
        return IQ_memory
    
    
    def generate_extreme_IQ(self, shots: int, p_ambig: float, 
                            noise_list: list, verbose = False) -> list:
        """Generates IQ data for the given logical state.
        Args:
            shots (int): Number of shots.
            p_ambig (float): Probability of ambiguous measurement.
            noise_list (list): List of noise parameters [two-qubit-fidelity, reset error, measurement error, idle error].
            logical (int, optional): Logical state. Defaults to None.
        """
        noise_list = [1e-8, 1e-8, 1e-8, 1e-8] if noise_list is None else noise_list
        IQ_dict = self.generate_IQ_dict()
        self.get_stim_circuit(noise_list)
        counts = self.get_counts(shots, self.stim_circ, verbose) # hardcoded for logical 0
        IQ_memory = self.counts_to_IQ_extreme(p_ambig, IQ_dict, counts)
        return IQ_memory




        
        
