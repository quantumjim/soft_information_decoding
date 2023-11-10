import numpy as np
import json
import random 
from soft_info import counts_to_det_syndr  


def generate_random_string(length):
    """Generates a random binary string of a given length."""
    return ''.join(random.choice(['0', '1']) for _ in range(length))

def generate_test_case(num_rounds, string_length):
    """Generates a test case with a specified number of measurement rounds and string length."""
    # Generate count string with one additional bit
    count_str = generate_random_string(string_length + 1)
    # Generate check strings
    check_str_parts = [generate_random_string(string_length) for _ in range(num_rounds)]
    # Combine count string and check strings, count string first
    input_str = " ".join([count_str] + check_str_parts)
    _resets = random.choice([True, False])
    expected_output = counts_to_det_syndr(input_str, _resets=_resets, verbose=False)
    return input_str, _resets, expected_output

def generate_test_cases(num_cases, max_rounds, max_string_length):
    """Generates multiple test cases with random rounds and string lengths."""
    test_cases = []
    for _ in range(num_cases):
        num_rounds = random.randint(1, max_rounds)  # Random number of rounds
        string_length = random.randint(1, max_string_length)  # Random string length
        test_case = generate_test_case(num_rounds, string_length)
        test_cases.append(test_case)
    return test_cases

def save_test_cases_to_file(test_cases, filename="tests/soft_info/PyMatching/test_cases_counts_to_detectors.json"):
    with open(filename, "w") as file:
        json.dump(test_cases, file, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# Example: Generate 10 test cases with up to 5 rounds and string lengths up to 10
test_cases = generate_test_cases(num_cases=500, max_rounds=10, max_string_length=15)
#print(test_cases)
save_test_cases_to_file(test_cases)
