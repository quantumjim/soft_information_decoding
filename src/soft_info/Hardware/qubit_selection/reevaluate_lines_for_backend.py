# import argparse
# from typing import List

# parser = argparse.ArgumentParser(
#     "reevaluate_lines_for_backend.py",
#     description="Revaluate the fidelity scores for previously saved chains.",
# )
# parser.add_argument(
#     "backends",
#     type=str,
#     help="The name of the IBM backend.",
#     nargs="*",
# )
# parser.add_argument(
#     "-a",
#     "--all",
#     help="Reevaluate for all previously evaluated backends.",
#     action="store_true",
# )
# parser.add_argument(
#     "n_qubits",
#     type=int,
#     nargs="*",
#     help="Sequence of integers for the lengths of the best chains to reevaluate.",
# )
# parser.add_argument(
#     "--not-symmetric",
#     action="store_false",
#     help="Consider the coupling map to be directional/not symmetric. "
#     "This may result in no suitable chains.",
# )
# parser.add_argument("-s", "--save", action="store_false", help="Save new scores.")

# args = parser.parse_args()

# import numpy as np

# from qiskit.transpiler import CouplingMap
# from qiskit_ibm_provider import IBMProvider

# from dissipative_quantum_circuits.reproducible import (
#     Experiment,
#     ExperimentFactory,
#     FieldMatch,
# )
# from dissipative_quantum_circuits.utils.qubit_selection import EvaluateFidelity
# from dissipative_quantum_circuits.utils.visualisation import Loader

# backend_names = args.backends
# all_backends = args.all
# n_qubits = args.n_qubits
# symmetric = args.not_symmetric
# save = args.save

# if all_backends:
#     backend_names = np.unique(
#         Experiment.preview_metadata("BestQubits")["backend"].tolist()
#     )

# if len(backend_names) == 0:
#     print("No backends provided")
#     exit()

# for backend_name in backend_names:
#     print("Retrieving backend with name='{}'.".format(backend_name))
#     backend = IBMProvider().get_backend(backend_name)

#     if symmetric:
#         chain_type = "symmetric"
#     else:
#         chain_type = "directed"
#     loader_description = "Reevaluating {} chain of {} qubits for {}..."

#     reevaluator_results = []
#     n_qubit_set = []

#     coupling_map = CouplingMap(backend.configuration().coupling_map)
#     if symmetric:
#         coupling_map.make_symmetric()

#     experiments = ExperimentFactory()
#     load_options = dict(
#         columns_to_exclude=["qubits", "score"],
#         backend=backend_name,
#         symmetric=symmetric,
#     )
#     if len(n_qubits) != 0:
#         load_options["n_qubits"] = FieldMatch("set", *n_qubits)

#     fidelity_evaluator = EvaluateFidelity(backend)
#     coupling_edges = coupling_map.get_edges()
#     with Loader(
#         desc="Reevaluating {} chains for {}...".format(chain_type, backend_name),
#         end="Completed evaluating scores for chains for {}.".format(backend_name),
#     ) as loader:
#         for exp in Experiment.load(
#             "BestQubits", latest=True, latest_method="metadata", **load_options
#         ):
#             qubits = exp["qubits"]
#             _n_qubits = exp["n_qubits"]
#             old_score = exp["score"]
#             new_score = fidelity_evaluator(
#                 qubits,
#                 coupling_edges,
#             )
#             if save:
#                 exp["score"] = new_score
#                 experiments.add_experiment(exp, _n_qubits)

#             loader.desc = loader_description.format(chain_type, _n_qubits, backend_name)
#             reevaluator_results.append(
#                 (qubits, old_score, new_score, exp[Experiment.DATE_SAVED])
#             )
#             n_qubit_set.append(_n_qubits)
#     if save:
#         experiments.save(overwrite=False)

#     for sub_n_qubits, (chain, old_score, new_score, date_saved) in zip(
#         n_qubit_set, reevaluator_results
#     ):
#         heading = "Best {} chain of {} qubits on {}".format(
#             chain_type, sub_n_qubits, backend_name
#         )
#         print(heading)
#         print("=" * len(heading))
#         print(" - Date: {:%Y-%m-%d %H:%M}".format(date_saved))
#         print(" - Old Score: {}".format(old_score))
#         print(" - New Score: {}".format(new_score))
#         print(" - Qubits:")
#         print("")
#         print("\t" + " -- ".join(["{:02}".format(qubit) for qubit in chain]))
#         print(" ")
