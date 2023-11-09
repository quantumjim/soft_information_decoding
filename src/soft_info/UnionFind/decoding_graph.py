# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-17

from copy import deepcopy
import warnings
from typing import List, Dict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .probabilities import llh_ratio
from ..Hardware.transpile_rep_code import get_repcode_IQ_map 


# TODO add correct a priori error rates to the graph
# TODO implement a hard flip edge
def soft_reweight(decoder, IQ_data, kde_dict: Dict, scaler_dict : Dict, layout : List[int], p_data : float = None, p_meas : float = None, common_measure = None):
    """Reweight the edges of a graph according to the log-likelihood ratio of the IQ datapoints & the a priori error rates.

    Args:
        decoder: The decoder object containing the decoding graph.
        IQ_data (list): The IQ data for multiple shots.
        kde_dict (dict): Dictionary mapping qubit index to a tuple containing its KDEs for state 0 and 1.
        scaler_dict (dict): Dictionary mapping qubit index to its scaler for normalization.
        p_data (float, optional): A priori error rate for data qubits.
        p_meas (float, optional): A priori error rate for measurement qubits.

    Returns:
        DecodingGraph: The reweighted PyGraph.
    """

    graph = deepcopy(decoder.decoding_graph.graph)

    p_data = p_data if p_data is not None else 6.836e-3  # Sherbrooke median
    p_meas = p_meas if p_meas is not None else 0

    tot_nb_checks = decoder.code.d - 1  # hardcoded for RepetitionCode

    #if layout is not None:
        #qubit_mapping = get_repcode_IQ_map(layout, decoder.code.T)

    for edge_idx, edge in enumerate(graph.edges()):
        src_node_idx, tgt_node_idx = graph.edge_list()[edge_idx]
        src_node, tgt_node = graph.nodes()[src_node_idx], graph.nodes()[
            tgt_node_idx]
        time_source, time_tgt = src_node['time'], tgt_node['time']

        if edge.qubits is not None: # data edges & mixed edges
            if time_source != time_tgt: # mixed edges
                p_mixed = p_data # find a better ratio! /5, /50 is worse for dist 10
                edge.weight = -np.log(p_mixed/(1-p_mixed))
            else: # data edges
                edge.weight = -np.log(p_data/(1-p_data))

        else: # time edges
            edge.weight = 0  #TODO -np.log(p_meas/(1-p_meas))

            

            # if time_source is None or time_tgt is None:
            #     continue # skip boundary edges

            if time_source > time_tgt: #TODO check if needed 
                src_node, tgt_node = tgt_node, src_node
                time_source, time_tgt = time_tgt, time_source
                src_node_idx, tgt_node_idx = tgt_node_idx, src_node_idx
                warnings.warn("time_source > time_tgt. Reordering them...") # For debugging purposes.

            if time_source != time_tgt: # only time edges but sanity check
                link_qubit_number = src_node.index # gives the index of the stabilizer measurement => the link qubit number
                IQ_point = IQ_data[src_node_idx]
                # IQ_point = IQ_data[time_source * tot_nb_checks + link_qubit_number]

                #TODO Wrong qubit into qubit_mapping?
                # layout_qubit_idx = qubit_mapping[src_node_idx] #TODO does src_node_idx start at 0 and goes to the end of the iq data?


                layout_qubit_idx = layout[link_qubit_number] #TODO clean up not needed parts!
                kde_0, kde_1 = kde_dict.get(layout_qubit_idx, (None, None))
                scaler = scaler_dict.get(layout_qubit_idx, None)

                weight = llh_ratio(IQ_point, kde_0, kde_1, scaler)

                edge.weight += weight

        # Round the weights to common measure
        if common_measure is not None:
            edge.weight = round(edge.weight/ common_measure) * common_measure

    return graph


def rx_draw_2D(graph, atypical_nodes=None):
    # deepopy the graph to not change it during plotting
    graph = deepcopy(graph)
    atypical_nodes_set = {(node['time'], node['index']) for node in atypical_nodes} if atypical_nodes is not None else set()
    fig, ax = plt.subplots()

    # First pass to find t_max and index_max
    t_max = float('-inf')
    index_max = float('-inf')
    for node in graph.nodes():
        time = node.time
        index = node.index

        if time is not None:
            t_max = max(t_max, time)
        index_max = max(index_max, index)

    # Second pass for plotting nodes
    for node_idx, node in enumerate(graph.nodes()):
        is_boundary = node.is_boundary
        time = node.time
        index = node.index
        if time is None:
            time = t_max / 2
            index = -0.5 if index == 0 else index_max + 0.5
            node.time = time
            node.index = index
        is_atypical = (node.time, node.index) in atypical_nodes_set
        if is_atypical:
            color = 'r'
        else:
            color = 'k' if is_boundary else 'b'
        ax.scatter(time, index, c=color, s=500 )

    # Plot edges
    for i, edge in enumerate(graph.edge_list()):
        src_node, tgt_node = edge
        edge_data = graph.edges()[i]
        weight = edge_data.weight
        qubits = edge_data.qubits

        src_time = graph.nodes()[src_node].time
        tgt_time = graph.nodes()[tgt_node].time
        src_index = graph.nodes()[src_node].index
        tgt_index = graph.nodes()[tgt_node].index

        edge_color = 'm'
        edge_label = f"{qubits} w:{weight:.2f}"

        if not qubits:  # Time edges
            edge_color = 'green'
            edge_label = f"w:{weight:.2f}"

        if src_time != tgt_time and src_index != tgt_index:  # Mixed edges
            edge_color = 'grey'
            edge_label = f"{qubits} w:{weight:.2f}"

        ax.plot([src_time, tgt_time], [src_index, tgt_index],
                color=edge_color, linestyle='--')

        mid_time = (src_time + tgt_time) / 2
        ax.text(mid_time, (src_index + tgt_index) / 2, edge_label)

    # Add legend in the top right corner, ensuring no overlap
    ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Boundary Qubits', markersize=10, markerfacecolor='black'),
                       plt.Line2D([0], [0], marker='o', color='w',
                                  label='Check Nodes', markersize=10, markerfacecolor='blue'),
                        plt.Line2D([0], [0], marker='o', color='w',
                                  label='Atypical nodes', markersize=10, markerfacecolor='orange'),          
                       plt.Line2D([0], [0], color='g',
                                  linestyle='--', label='Time edges'),
                       plt.Line2D([0], [0], color='grey',
                                  linestyle='--', label='Mixed edges'),
                       plt.Line2D([0], [0], color='m', linestyle='--', label='Qubit edges')],
              loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to ensure the legend does not overlap
    plt.tight_layout(rect=[0, 0, 2, 1])

    # Set y-axis ticks to half-integers and label them to represent qubit numbers
    y_ticks = [i - 0.5 for i in range(index_max + 2)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i) for i in range(index_max + 2)])

    ax.set_ylabel('Qubit Index')
    ax.set_xlabel('Syndrome Round')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.show()


# NOT NEEDED
def rx_draw(graph):
    G_nx = nx.Graph()

    # Dictionary to hold 'is_boundary' information for nodes
    is_boundary_dict = {}

    # Add nodes to NetworkX graph
    for node in graph.node_indexes():
        is_boundary = graph.nodes()[node]['is_boundary']
        is_boundary_dict[node] = is_boundary
        G_nx.add_node(node, color='red' if is_boundary else 'black')

    # Add edges to NetworkX graph
    for i, edge in enumerate(graph.edge_list()):
        src, tgt = edge
        edge_data = graph.get_edge_data(src, tgt)
        # Default weight to 1 if not available
        weight = edge_data.get('weight', 1)
        qubits = graph.edges()[i]['qubits']
        G_nx.add_edge(src, tgt, weight=weight, qubits=qubits)

    # Compute layout for positioning
    pos = nx.spring_layout(G_nx)

    # Draw graph
    node_colors = [G_nx.nodes[node]['color'] for node in G_nx.nodes()]
    nx.draw(G_nx, with_labels=True, node_color=node_colors)

    # Draw edge labels with weights and qubits
    edge_labels = {(src, tgt): f"w:{data['weight']}, {data['qubits']}" for (
        src, tgt, data) in G_nx.edges(data=True)}
    nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels)

    # Annotate nodes with 'is_boundary'
    for node, (x, y) in pos.items():
        if is_boundary_dict.get(node, False):
            plt.annotate("Boundary",
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         color='red')

    plt.show()
