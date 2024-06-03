import pymatching
import stim
import numpy as np
import json

from src import cpp_soft_info
from cpp_soft_info import counts_to_det_syndr

def get_matching():
    circuit = stim.Circuit.generated("repetition_code:memory",
                                     distance=2,
                                     rounds=1,
                                     after_clifford_depolarization=0.1)
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)

    return matching


def test_get_edges():
    matching = get_matching()
    assert matching.edges() == cpp_soft_info.get_edges(matching._matching_graph)


def test_add_edge():
    matching1, matching2 = get_matching(), get_matching()

    node1 = 0
    node2 = 1
    observables = {0, 1, 2}  # Replace with your set of observables
    weight = 1.0
    error_probability = 0.1
    merge_strategy = "replace"  # Replace with your desired merge strategy

    cpp_soft_info.add_edge(matching1._matching_graph, node1, node2,
                           observables, weight, error_probability, merge_strategy)
    matching2.add_edge(node1=node1, node2=node2, fault_ids=observables, weight=weight,
                       error_probability=error_probability, merge_strategy=merge_strategy)

    assert matching1.edges() == matching2.edges()


def test_add_boundary_edge():
    matching1, matching2 = get_matching(), get_matching()

    node1 = 0
    observables = {0, 1, 2}  # Replace with your set of observables
    weight = 1.0
    error_probability = 0.1
    merge_strategy = "replace"  # Replace with your desired merge strategy

    cpp_soft_info.add_boundary_edge(
        matching1._matching_graph, node1, observables, weight, error_probability, merge_strategy)
    matching2.add_boundary_edge(node=node1, fault_ids=observables, weight=weight,
                                error_probability=error_probability, merge_strategy=merge_strategy)
    
    assert matching1.edges() == matching2.edges()

def test_counts_to_det_syndr():
    # Example test cases
    test_cases = [
        # Format: (input_str, _resets, expected_output)
        ("010 01 10", True, [0, 1, 1, 1, 0, 1]),
        ("100 01 10", True, [0, 1, 1, 1, 1, 1]),
        ("110 00 11", False, [1, 1, 0, 0, 0, 1]),        
        ("000 00 00 00 00 00", True, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("000 00 00 00 00 00", False, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("000 11 11 11 11 11", False, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),        
    ]

    # Load test cases from the generated file
    filename = "tests/soft_info/PyMatching/test_cases_counts_to_detectors.json"
    with open(filename, "r") as file:
        generated_test_cases = json.load(file)

    # Convert the expected output in generated test cases back to numpy arrays
    for case in generated_test_cases:
        input_str, resets, expected_output = case
        expected_output = np.array(expected_output)
        test_cases.append((input_str, resets, list(expected_output)))
        
    for input_str, resets, expected in test_cases:
        # Call the function
        result = cpp_soft_info.counts_to_det_syndr(input_str, _resets=resets, verbose=False)
        
        # Check if the result matches the expected output
        assert np.array_equal(result, expected), f"Failed for input: {input_str} with resets={resets}"

