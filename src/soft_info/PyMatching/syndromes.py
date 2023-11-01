# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-11-01

import numpy as np

def count_string_to_syndromes(input_str, verbose=False):
    reversed_str = input_str[::-1]
    if verbose:
        print("Reversed str:", reversed_str)

    last_part = reversed_str.split(" ")[-1]
    if verbose:
        print("Count str:", last_part)

    xor_result = ''.join([str((int(last_part[i]) + int(last_part[i + 1])) % 2) for i in range(len(last_part) - 1)])
    if verbose:
        print("XOR result:", xor_result)

    first_part = ''.join(reversed_str.split(" ")[:-1])
    if verbose:
        print("First part:", first_part)
        
    numpy_list = np.array([int(bit) for bit in first_part]+ [int(bit) for bit in xor_result])
    if verbose:
        print("Numpy list:", numpy_list)
    
    return numpy_list