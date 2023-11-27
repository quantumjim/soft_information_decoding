import cpp_soft_info
import pymatching
import stim
import sys
sys.path.insert(
    0, r'/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/build')


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
