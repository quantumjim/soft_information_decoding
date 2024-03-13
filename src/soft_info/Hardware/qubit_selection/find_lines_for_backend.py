# import argparse

# parser = argparse.ArgumentParser("find_lines_for_backend.py")
# parser.add_argument("backend", type=str, help="The name of the IBM backend.")
# parser.add_argument(
#     "n_qubits",
#     type=int,
#     nargs="+",
#     help="Sequence of integers for the lengths of the best chains to identify.",
# )
# parser.add_argument(
#     "--not-symmetric",
#     action="store_false",
#     help="Consider the coupling map to be directional/not symmetric. "
#     "This may result in no suitable chains.",
# )

# args = parser.parse_args()

# from qiskit_ibm_provider import IBMProvider

# from dissipative_quantum_circuits.utils.qubit_selection import BackendEvaluator
# from dissipative_quantum_circuits.utils.visualisation import Loader

# backend_name = args.backend
# n_qubits = args.n_qubits
# symmetric = args.not_symmetric
# print("Retrieving backend with name='{}'.".format(backend_name))
# backend = IBMProvider().get_backend(backend_name)


# if symmetric:
#     chain_type = "symmetric"
# else:
#     chain_type = "directed"
# loader_description = "Evaluating best {} chain of {} qubits for {}..."

# evaluator_results = []
# evaluator = BackendEvaluator(backend, symmetric)

# with Loader(
#     desc=loader_description.format(chain_type, n_qubits[0], backend_name),
#     end="Completed evaluating best chains for {}.".format(backend_name),
# ) as loader:
#     for i_sub, sub_n_qubits in enumerate(n_qubits):
#         loader.desc = loader_description.format(chain_type, sub_n_qubits, backend_name)
#         evaluator_results.append(evaluator.evaluate(sub_n_qubits))


# from dissipative_quantum_circuits.reproducible import Experiment

# for sub_n_qubits, (best_chain, chain_score, num_subsets, chain_metadata) in zip(
#     n_qubits, evaluator_results
# ):
#     if best_chain is None:
#         heading = "Failed to find chain of {} qubits on {}".format(
#             sub_n_qubits, backend_name
#         )
#         print(heading)
#         print("=" * len(heading))
#         print("")
#     else:
#         heading = "Best {} chain of {} qubits on {}".format(
#             chain_type, sub_n_qubits, backend_name
#         )
#         print(heading)
#         print("=" * len(heading))
#         print(" - Score: {}".format(chain_score))
#         print(" - Number of subsets found: {}".format(num_subsets))
#         print(" - Best qubits:")
#         print(" - Metadata:")
#         for k, v in chain_metadata.items():
#             print("   - {}: {}".format(k, v))
#         print("")
#         print("\t" + " -- ".join(["{:02}".format(qubit) for qubit in best_chain]))
#         print(" ")

#         experiment = Experiment(
#             "BestQubits",
#             "BestChain",
#             description="Best N qubits forming a single continuous chain.",
#             n_qubits=sub_n_qubits,
#             score=chain_score,
#             backend=backend_name,
#             qubits=best_chain,
#             symmetric=symmetric,
#         )
#         experiment.save()


