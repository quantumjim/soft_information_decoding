# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-17

import numpy as np
from probabilities_functions import llh_ratio

def soft_reweight(graph, sorted_IQ_data, distr_0, distr_1):
    """Reweight the edges of a graph according to the log-likelihood ratio of the IQ datapoints.

    Args:
        graph (DecodingGraph: PyGraph): The graph to reweight.
        sorted_IQ_data (dict): The IQ data for multiple shots.
        distr_0 (scipy.stats distribution): The IQ distribution for outcome = 0.
        distr_1 (scipy.stats distribution): The IQ distribution for outcome = 1.

    Returns:
        DecodingGraph: The reweighted PyGraph.
    """
    # TODO: reweight the diagonal edges & the data edges

    IQ_point = sorted_IQ_data[edge['qubits']]
    tot_nb_checks = max(graph.nodes()[node]['index']
                        for node in graph.node_indexes()) + 1

    for i, edge in enumerate(graph.edges()):

        if edge['qubits'] == None:  # time edges
            u = graph.edge_list(i)[0]  # the first node has a lower time
            link_qubit_number = graph.nodes()[u]['index']
            synd_round = graph.nodes()[u]['time']
            IQ_point = sorted_IQ_data[synd_round *
                                      tot_nb_checks + link_qubit_number]
            weight = - llh_ratio(IQ_point, distr_0, distr_1)
            edge['weight'] += weight # add it up since other error as well

    return graph
