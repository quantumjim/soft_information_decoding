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

    def generate_IQ(self, shots: int, noise_model: PauliNoiseModel, logical: int) -> dict:
        counts = self.get_counts(shots, noise_model, logical)
        IQ_memory = self.counts_to_IQ(counts)
        return IQ_memory




        
        
