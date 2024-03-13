import argparse

def main():
    parser = argparse.ArgumentParser("find_lines_for_backend.py")
    parser.add_argument("backend", type=str, help="The name of the IBM backend.")
    parser.add_argument(
        "n_qubits",
        type=int,
        nargs="+",
        help="Sequence of integers for the lengths of the best chains to identify.",
    )
    parser.add_argument(
        "--not-symmetric",
        action="store_false",
        help="Consider the coupling map to be directional/not symmetric. "
        "This may result in no suitable chains.",
    )

    args = parser.parse_args()

    from qiskit_ibm_provider import IBMProvider

    from dissipative_quantum_circuits.utils.qubit_selection import BackendEvaluator
    from dissipative_quantum_circuits.utils.visualisation import Loader

    backend_name = args.backend
    n_qubits = args.n_qubits
    symmetric = args.not_symmetric
    print(f"Retrieving backend with name='{backend_name}'.")
    backend = IBMProvider().get_backend(backend_name)

    if symmetric:
        chain_type = "symmetric"
    else:
        chain_type = "directed"
    loader_description = f"Evaluating best {chain_type} chain of {{}} qubits for {backend_name}..."

    evaluator_results = []
    evaluator = BackendEvaluator(backend, symmetric)

    with Loader(
        desc=loader_description.format(n_qubits[0]),
        end=f"Completed evaluating best chains for {backend_name}.",
    ) as loader:
        for i_sub, sub_n_qubits in enumerate(n_qubits):
            loader.desc = loader_description.format(sub_n_qubits)
            evaluator_results.append(evaluator.evaluate(sub_n_qubits))

    from dissipative_quantum_circuits.reproducible import Experiment

    for sub_n_qubits, (best_chain, chain_score, num_subsets, chain_metadata) in zip(
        n_qubits, evaluator_results
    ):
        if best_chain is None:
            heading = f"Failed to find chain of {sub_n_qubits} qubits on {backend_name}"
            print(heading)
            print("=" * len(heading))
            print("")
        else:
            heading = f"Best {chain_type} chain of {sub_n_qubits} qubits on {backend_name}"
            print(heading)
            print("=" * len(heading))
            print(f" - Score: {chain_score}")
            print(f" - Number of subsets found: {num_subsets}")
            print(" - Best qubits:")
            print(" - Metadata:")
            for k, v in chain_metadata.items():
                print(f"   - {k}: {v}")
            print("")
            print("\t" + " -- ".join([f"{qubit:02}" for qubit in best_chain]))
            print(" ")

            experiment = Experiment(
                "BestQubits",
                "BestChain",
                description="Best N qubits forming a single continuous chain.",
                n_qubits=sub_n_qubits,
                score=chain_score,
                backend=backend_name,
                qubits=best_chain,
                symmetric=symmetric,
            )
            experiment.save()

if __name__ == "__main__":
    main()
