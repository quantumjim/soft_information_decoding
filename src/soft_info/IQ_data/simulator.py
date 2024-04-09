# Maurice D. Hanisch 
# Created 18.11.2023


import warnings
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import stim

from ..Hardware.transpile_rep_code import get_repcode_layout, get_repcode_IQ_map
from ..Hardware.backend_noise import get_avgs_from_dict, get_noise_dict_from_backend
from ..Hardware.qubit_selector import BackendEvaluator
from ..Probabilities.KDE import get_KDEs
from ..Stim_circuits.circuits import RepetitionCodeStimCircuit
from Scratch import load_calibration_memory



class RepCodeIQSimulator():

    _kde_cache = OrderedDict()
    _max_cache_size = 4

    def __init__(self, provider, d: int, T: int, device: int, _is_hex: bool = True, 
                 other_date=None, double_msmt: bool = False, best_path: bool = False) -> None:
        self.provider = provider

        self.d = d
        self.T = T

        self.device = device
        self.other_date = other_date
        self.backend = self.provider.get_backend(self.device)

        if best_path:
            raise NotImplementedError("Best path not implemented yet, because of the date parameter")
            evaluator = BackendEvaluator(self.backend)
            longest_path, _, _, path_info = evaluator.find_longest_good_RepCode_string(Ancilla_threshold=0.3) # as the device runs!
            self.layout = longest_path[1::2] + longest_path[0::2]
            self.path_info = path_info
        else:
            self.layout = get_repcode_layout(self.d, self.backend, _is_hex=_is_hex) if self.device != 'ibm_torino' else RepCodeIQSimulator.get_layout_torino(self.d)
        
        self.qubit_mapping = get_repcode_IQ_map(self.layout, self.T)
        self.stim_circ = None

        kde_cache_key = (device, other_date, double_msmt)

        # Attempt to retrieve from cache
        if kde_cache_key in RepCodeIQSimulator._kde_cache:
            cached = RepCodeIQSimulator._kde_cache[kde_cache_key]
            self.kde_dict, self.scaler_dict, self.all_memories, self.noise_dict= cached['kde'], cached['scaler'], cached['memories'], cached['noise']
            self.msmt_err_dict = cached.get('msmt_error', {})
            RepCodeIQSimulator._kde_cache.move_to_end(kde_cache_key)
        else:
            # If not in cache, load or calculate the data
            self.noise_dict = get_noise_dict_from_backend(provider, device, date=other_date)
            if double_msmt:
                self.all_memories, self.gmm_dict, self.msmt_err_dict = load_calibration_memory(provider, tobecalib_backend=device, 
                                                                                               other_date=other_date, double_msmt=double_msmt,
                                                                                               post_process=True)
            else:
                self.all_memories = load_calibration_memory(provider, tobecalib_backend=device, other_date=other_date)
                self.msmt_err_dict = {}
            self.kde_dict, self.scaler_dict = get_KDEs(self.all_memories)
            

            # Update cache with new entry
            RepCodeIQSimulator._update_cache(kde_cache_key, {
                'kde': self.kde_dict, 'scaler': self.scaler_dict, 'memories': self.all_memories, 'noise': self.noise_dict,
                'msmt_error': self.msmt_err_dict
            })

        # Prepare noise list based on the newly loaded or cached noise_dict
        self._prepare_noise_list(double_msmt)

    def _prepare_noise_list(self, double_msmt):
        self.noise_avgs = get_avgs_from_dict(self.noise_dict, self.layout)
        self.noise_list = [self.noise_avgs['two_gate'], self.noise_avgs['single_gate'], self.noise_avgs["t1_err"], 
                           self.noise_avgs["t2_err"]]
        if double_msmt:
            p_soft, p_hard = 0, 0
            for value in self.msmt_err_dict.values():
                p_soft += value['p_soft']
                p_hard += value['p_hard']
            p_soft /= len(self.msmt_err_dict)
            p_hard /= len(self.msmt_err_dict)
            self.noise_list += [(p_soft+p_hard), p_hard, p_soft]
            self.noise_list[-1] = 0 # for simulating because soft comes from IQ
        else:
            msmt_err = self.noise_avgs['readout']
            self.noise_list += [msmt_err, msmt_err*2/3, msmt_err/3]
            self.noise_list[-1] = 0 # for simulating because soft comes from IQ

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
    def _update_cache(key, value):
        cache = RepCodeIQSimulator._kde_cache
        if len(cache) >= RepCodeIQSimulator._max_cache_size:
            cache.popitem(last=False)  # Remove the oldest item
        cache[key] = value


    def get_stim_circuit(self, logical: str, xbasis: bool, resets: bool, individual: bool = False,):
        if not individual:
            code = RepetitionCodeStimCircuit(self.d, self.T, xbasis, resets, 
                                             noise_list=self.noise_list)
        else:
            code = RepetitionCodeStimCircuit(self.d, self.T, xbasis, resets,
                                             noise_list=self.noise_list, layout=self.layout,
                                             msmt_err_dict=self.msmt_err_dict)
        self.stim_circ = code.circuits[logical]

        return code.circuits[logical]
        
    def get_counts(self, shots: int, stim_circuit: stim.Circuit, resets: bool, verbose=False) -> dict:
        meas_outcomes = stim_circuit.compile_sampler(seed=42).sample(shots)
        # print("generated stim counts")
        counts = {}
        for row in meas_outcomes:
            count_str = ''
            for nb, bit in enumerate(row):
                count_str += '0' if bit == False else '1'
                if (nb+1) % (self.d-1) == 0 and nb < self.T*(self.d-1):
                    count_str += ' ' 
            count_str = count_str[::-1]
            if count_str in counts:
                counts[count_str] += 1
            else:
                counts[count_str] = 1
        
        print("not correcting the counts for no resets")
        # DO NOT CORRECT FOR NO RESETS BECAUSE STIM CIRCUIT ALREADY DOES THIS!
        # # correct samples to have no reset counts if _resets == False
        # if resets == False:
        #     no_reset_counts = {}
        #     for count_key, shots in counts.items():
        #         parts=count_key.split(" ")
        #         print("parts:", parts) if verbose else None
        #         count_part = parts[0]
        #         print("count_part:", count_part) if verbose else None  
        #         check_parts = parts[1:]
        #         print("check_parts:", check_parts, "\n") if verbose else None

        #         for i in range(len(check_parts)):
        #             if i == 0:
        #                 print("skipped last check part:", check_parts[-1], "\n") if verbose else None
        #                 continue
        #             current_check_str = check_parts[-(i+1)]
        #             print("current_check_str:", current_check_str) if verbose else None
        #             prev_check_str = check_parts[-i]
        #             print("prev_check_str:", prev_check_str) if verbose else None
        #             new_check_str = ''
        #             for bit1, bit2 in zip(prev_check_str, current_check_str):
        #                 new_check_str += str(int(bit1)^int(bit2))
        #             print("new_check_str:", new_check_str, "\n") if verbose else None
        #             check_parts[-(i+1)] = new_check_str

        #         print("\ncheck_parts after modulo:", check_parts) if verbose else None

        #         new_count_str = count_part + " " +  ' '.join(check_parts)
        #         print("\nnew_count_str:", new_count_str) if verbose else None

                
        #         if new_count_str in no_reset_counts:
        #             no_reset_counts[new_count_str] += shots
        #         else:
        #             no_reset_counts[new_count_str] = shots
        #     counts = no_reset_counts
        
        # print("finished correcting counts")
        return counts


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


    def generate_IQ(self, shots: int, logical: str, xbasis: bool, resets: bool, individual: bool = False, verbose = False) -> list:
        """Generates IQ data for the given logical state.
        Args:
            shots (int): Number of shots.
            noise_list (list): List of noise parameters [two-qubit-fidelity, reset error, measurement error, idle error].
            logical (int, optional): Logical state. Defaults to None.
        """
        self.get_stim_circuit(logical, xbasis, resets, individual)
        counts = self.get_counts(shots, self.stim_circ, resets, verbose)
        IQ_memory = self.counts_to_IQ(counts, verbose)
        
        return IQ_memory
    


        
        
