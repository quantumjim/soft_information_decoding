from src import cpp_soft_info

import stim
import pymatching
import numpy as np


def test_processGraph_test():
    circuit = stim.Circuit.generated("repetition_code:memory",
                                 distance=2,
                                 rounds=1,
                                 after_clifford_depolarization=0.1)

    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)

    try: 
        cpp_soft_info.processGraph_test(matching._matching_graph)
    except TypeError:
        raise AssertionError("processGraph_test() raised TypeError, check compatibility with PyMatching")
    
def test_reweight_edges_to_one():
    circuit = stim.Circuit.generated("repetition_code:memory",
                                 distance=2,
                                 rounds=1,
                                 after_clifford_depolarization=0.1)

    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)

    cpp_soft_info.reweight_edges_to_one(matching._matching_graph)

    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        assert edge_data['weight'] == 1

def test_reweight_edges_informed():
    d = 3
    p_data = 0.4
    p_mixed = 0.2
    p_meas = 0.5
    tolerance = 1e-8

    circuit = stim.Circuit.generated("repetition_code:memory",
                                     distance=d,
                                     rounds=d,
                                     after_clifford_depolarization=0.1)

    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)

    cpp_soft_info.reweight_edges_informed(matching._matching_graph, distance=d, p_data=p_data, p_mixed=p_mixed, p_meas=p_meas)

    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        if tgt_node is None:  # Boundary
            assert np.isclose(edge_data['weight'], -np.log(p_data / (1 - p_data)), atol=tolerance)
        elif tgt_node == src_node + 1:  # Data
            assert np.isclose(edge_data['weight'], -np.log(p_data / (1 - p_data)), atol=tolerance)
        elif tgt_node == src_node + (d - 1):  # Time
            assert np.isclose(edge_data['weight'], -np.log(p_meas / (1 - p_meas)), atol=tolerance)
        elif tgt_node == src_node + (d - 1) + 1:  # Mixed
            assert np.isclose(edge_data['weight'], -np.log(p_mixed / (1 - p_mixed)), atol=tolerance, rtol=0)