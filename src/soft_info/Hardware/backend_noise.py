# Maurice Hanisch mhanisc@ethz.ch
# Created 2024-01-13

import datetime as datetime
from dateutil import parser
import pytz

import numpy as np


def get_noise_dict_from_backend(provider, device: str, used_qubits: list = None, date: str = None):
    """Get the noise dictionary from the backend.
    
    Args:
        backend (qiskit backend): Needed backend.
        layout (list): List of qubits to get noise from (use layout directly).
    """
    backend = provider.get_backend(device)
    if used_qubits is None:
        used_qubits = list(range(backend.configuration().n_qubits))
        # print("Used qubits not specified, using all qubits:", used_qubits)
    noise_dict = {} 
    round_time = 4000e-9 # HARCODED mean for (10, 10) to (50, 50) RepCodes 
    if date is not None and type(date) is str:
        date = parser.parse(date)
        date = date.astimezone(pytz.utc)

    properties = backend.properties(datetime=date)
    for qubit in used_qubits:
        noise_dict[qubit] = {}

        noise_dict[qubit]['readout_error'] = properties.readout_error(qubit)
        noise_dict[qubit]['gate'] = np.mean([properties.gate_error(gate, qubit) for gate in ["sx", "id", "x"]])
    
        noise_dict[qubit]['T1'] = properties.t1(qubit)
        noise_dict[qubit]['T2'] = properties.t2(qubit)

        idle_error = 1 - np.exp(-round_time / min(properties.t1(qubit), properties.t2(qubit)))
        noise_dict[qubit]['idle'] = idle_error

        noise_dict[qubit]['2-gate'] = {}

    for pair in backend.coupling_map:
        if pair[0] in used_qubits and pair[1] in used_qubits:
            two_gate_error = properties.gate_error('ecr', pair)
            if two_gate_error > .5:
                two_gate_error = .5
            noise_dict[pair[0]]['2-gate'][pair[1]] = two_gate_error
            noise_dict[pair[1]]['2-gate'][pair[0]] = two_gate_error

    return noise_dict


def get_avgs_from_dict(noise_dict: dict, used_qubits: list):
    """Get the averages from the noise dictionary.
    
    Returns:
        list: Dict of averages in order [idle, readout, single_gate, two_gate]
    """

    idle_avg = np.average([noise_dict[qubit]['idle'] for qubit in noise_dict if qubit in used_qubits])
    readout_avg = np.average([noise_dict[qubit]['readout_error'] for qubit in noise_dict if qubit in used_qubits])
    single_gate_avg = np.average([noise_dict[qubit]['gate'] for qubit in noise_dict if qubit in used_qubits])
    two_gate_avg = np.average([noise_dict[qubit]['2-gate'][connection]
                            for qubit in noise_dict if qubit in used_qubits
                            for connection in noise_dict[qubit]['2-gate'] if connection in used_qubits])
    
    return {'idle': idle_avg,
            'readout': readout_avg,
            'single_gate': single_gate_avg,
            'two_gate': two_gate_avg}
 





