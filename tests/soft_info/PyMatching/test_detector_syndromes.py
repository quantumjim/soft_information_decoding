import numpy as np
import json

from src.soft_info.PyMatching.detector_syndromes import counts_to_det_syndr  

def test_counts_to_det_syndr():
    # Example test cases
    test_cases = [
        # Format: (input_str, _resets, expected_output)
        ("010 01 10", True, np.array([0, 1, 1, 1, 0, 1])),
        ("100 01 10", True, np.array([0, 1, 1, 1, 1, 1])),
        ("110 00 11", False, np.array([1, 1, 0, 0, 0, 1])),        
        ("000 00 00 00 00 00", True, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("000 00 00 00 00 00", False, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("000 11 11 11 11 11", False, np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])),        
    ]

    # Load test cases from the generated file
    filename = "tests/soft_info/PyMatching/test_cases_counts_to_detectors.json"
    with open(filename, "r") as file:
        generated_test_cases = json.load(file)

    # Convert the expected output in generated test cases back to numpy arrays
    for case in generated_test_cases:
        input_str, resets, expected_output = case
        expected_output = np.array(expected_output)
        test_cases.append((input_str, resets, expected_output))

    for input_str, resets, expected in test_cases:
        # Call the function
        result = counts_to_det_syndr(input_str, _resets=resets, verbose=False)
        
        # Check if the result matches the expected output
        assert np.array_equal(result, expected), f"Failed for input: {input_str} with resets={resets}"


