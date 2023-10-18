# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-17

import numpy as np
from probabilities_functions import llh_ratio


def soft_reweight(graph, IQ_data, distr_0, distr_1, p_data=None, p_meas=None):
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
    # TODO: take the coupling map and update the weights accordingly

    p_data = p_data if p_data is not None else 6.836e-3  # Sherbrooke median
    p_meas = p_meas if p_meas is not None else 0

    tot_nb_checks = max(graph.nodes()[node]['index']
                        for node in graph.node_indexes()) + 1

    for i, edge in enumerate(graph.edges()):
        if edge['qubits'] != None:
            edge['weight'] = p_data/(1-p_data)
        else:
            edge['weight'] = p_meas/(1-p_meas)

        node_1, node_2 = graph.nodes()[graph.edge_list(i)[0]], graph.nodes()[
            graph.edge_list(i)[0]]
        time_1, time_2 = node_1['time'], node_2['time']
        if time_1 > time_2:
            node_1, node_2 = node_2, node_1
            time_1, time_2 = time_2, time_1
        if time_1 != time_2:
            link_qubit_number = node_1['index']
            IQ_point = IQ_data[time_1 *
                               tot_nb_checks + link_qubit_number]
            weight = llh_ratio(IQ_point, distr_0, distr_1)
            edge['weight'] += weight  # sum for diagonal edges

        edge['weight'] = np.log(edge['weight'])
        
    return graph
