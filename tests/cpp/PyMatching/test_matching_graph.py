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
    