# Maurice D. Hanisch mhanisc@ethz.ch
# 2024-03-23

def generate_subsets_with_center(D, d, verbose=False):
    """Generate the least possible subsets of size d from a list of size D with a center subset so that the list is covered, starting from 0.
    
    Args:
        D (int): The size of the list.
        d (int): The size of the subsets.

    Returns:
        list: A list of subsets of size d.
    
    Example:
        D = 7 -> [0, 1, 2, 3, 4, 5, 6]
        d = 3
        generate_subsets_with_center(D, d) -> [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    """
    if verbose:
        print(f"D = {D}, d = {d}")

    if D == d: 
        return [list(range(D))]
    
    Q = (D // 2 - d) // (d - 1)
    R = D - 2 * d - 2 * Q * (d - 1)
    if verbose:
        print(f"Q = {Q}, R = {R}")

    if R < (d - 1):
        C_start = D // 2 - d // 2
        center_subset = list(range(C_start, C_start + d))
    else:
        center_subset = None
        Q += 1
    if verbose:
        print(center_subset)

    L_subsets = [list(range(0, d))]
    R_subsets = [list(range(D - d, D))]
    
    for i in range(Q):
        L_start = d - 1 + i * (d - 1)
        L_subsets.append(list(range(L_start, L_start + d)))
        R_end = D - (d - 1) - i * (d - 1)
        R_subsets.append(list(range(R_end - d, R_end)))
    
    R_subsets.reverse()

    if verbose:
        print(f"L_subsets = {L_subsets}")
        print(f"R_subsets = {R_subsets}")

    if center_subset:
        subsets = L_subsets + [center_subset] + R_subsets
    else:
        subsets = L_subsets + R_subsets

    if verbose:
        print(f"subsets = {subsets}")
    
    return subsets

def get_cols_to_keep(subset, T, D):
    """
    Args: 
        subset (list): A list of indices of the subset 
        T (int): The number of syndrome rounds
        D (int): The distance of the big code
    """
    cols_to_keep = []
    for i in range(T):
        a_cols_to_keep = [x + i * (D-1) for x in subset[:-1]]
        cols_to_keep.extend(a_cols_to_keep)
    c_cols_to_keep = [T * (D-1) + x for x in subset]
    cols_to_keep.extend(c_cols_to_keep)

    return cols_to_keep


def get_subsample_layout(subset, link_qubits, code_qubits):
    """
    Args:
        subset (list): A list of indices of the subset
        link_qubits (int): The number of link qubits
        code_qubits (int): The number of code qubits

    Returns:
        layout (list): the layout = link_qubits + code_qubits
    """
    new_links = [link_qubits[i] for i in subset[:-1]]
    new_codes = [code_qubits[i] for i in subset]
    layout = new_links + new_codes

    return layout