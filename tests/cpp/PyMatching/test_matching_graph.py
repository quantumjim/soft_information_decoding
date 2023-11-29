import cpp_soft_info

import stim
import pymatching


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

def test_reweight_edges_to_one_diag():
    circuit = stim.Circuit.generated("repetition_code:memory",
                                 distance=2,
                                 rounds=1,
                                 after_clifford_depolarization=0.1)

    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)

    cpp_soft_info.reweight_edges_to_one_diag(matching._matching_graph, p_mixed=0.5, distance=2)

    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        if tgt_node == None:
            assert edge_data['weight'] == 1
        if tgt_node == src_node + 2:
            assert edge_data['weight'] == 0.5
        if tgt_node == src_node + 1:
            assert edge_data['weight'] == 1
