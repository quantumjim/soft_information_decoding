# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-11-01

import numpy as np


def counts_to_det_syndr(input_str, _resets = False, verbose=False):
    # Step 1: Reverse the input string
    reversed_str = input_str[::-1]
    if verbose:
        print("Reversed str:", reversed_str)

    # Step 2: Separate the count string
    split_str = reversed_str.split(" ")
    count_str = split_str[-1]
    check_str = split_str[:-1]
    
    # The number of measurement rounds T is the number of parts in the check_str
    T = len(check_str)
    
    # Check if the count string is one bit longer than the following strings
    if not all(len(count_str) == len(check_part) + 1 for check_part in check_str):
        raise ValueError("Count string must be one bit longer than each check string.")
    
    if verbose:
        print("Count str:", count_str)
        print("Check str:", check_str)
        print("Number of measurement rounds (T):", T)
    
    # Step 3: Separate the check_str into T parts
    # check_str_parts = [''.join(check_str[i::T]) for i in range(T)] NOT NEEDED!
    check_str_parts = check_str
    if verbose:
        print("Check str part T:", check_str_parts)
    
    # Step 4: Initialize detector string list with T+1 empty strings
    detector_str_parts = [''] * (T+1)
    
    # Step 5: Set the first part of the detector string
    detector_str_parts[0] = check_str_parts[0]

    if not _resets:
        # Step 5.5: No resets -> compute the synd meas outcomes
        meas_str_parts = [''] * (T+1)
        meas_str_parts[0] = check_str_parts[0]
        for i in range(1, T):
            meas_str_parts[i] = ''.join(
                str((int(check_str_parts[i-1][j]) + int(check_str_parts[i][j])) % 2) 
                for j in range(len(check_str_parts[i]))
            )
        if verbose:
            print("_reset = False: Syndrome meas outcomes:", meas_str_parts)
    
        # Step 6: Compute parts 2 to T of the detector string
        for i in range(1, T):
            detector_str_parts[i] = ''.join(
                str((int(meas_str_parts[i-1][j]) + int(meas_str_parts[i][j])) % 2) 
                for j in range(len(meas_str_parts[i]))
            )
        if verbose:
            print("_reset = False: Detector str parts:", detector_str_parts)
    
    else:
        for i in range(1, T):
            detector_str_parts[i] = ''.join(
                str((int(check_str_parts[i-1][j]) + int(check_str_parts[i][j])) % 2) 
                for j in range(len(check_str_parts[i]))
            )
        if verbose:
            print("_reset = True: Detector str parts:", detector_str_parts)
    
    
    
    # Step 7: Compute the XOR string from the count string
    xor_result = ''.join(
        str((int(count_str[i]) + int(count_str[i + 1])) % 2) 
        for i in range(len(count_str) - 1)
    )
    if verbose:
        print("XOR result:", xor_result)
    
    # Compute the (T+1)th part of the detector string using the T-1 part of the check str
    if not _resets:
        if verbose:
            print("_reset = False: meas_str_parts[T-1]:", meas_str_parts[T-1])
        detector_str_parts[T] = ''.join(
            str((int(xor_result[j]) + int(meas_str_parts[T-1][j])) % 2) 
            for j in range(len(xor_result))
        )
    
    else:
        detector_str_parts[T] = ''.join(
            str((int(xor_result[j]) + int(check_str_parts[T-1][j])) % 2) 
            for j in range(len(xor_result))
        )
    
    if verbose:
        print("Detector str after count detector:", detector_str_parts)
    
    # Convert detector string parts to a single numpy array
    numpy_list = np.array([int(bit) for part in detector_str_parts for bit in part])
    
    # if verbose:
    #     print("Numpy list:", numpy_list)
    
    return numpy_list