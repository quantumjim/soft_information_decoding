# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-17

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .probabilities import llh_ratio


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
        if edge['qubits'] is not None:
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

        edge['weight'] = -np.log(edge['weight'])

    return graph


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


def rx_draw_2D(graph):
    fig, ax = plt.subplots()

    node_info_dict = {}
    t_max = float('-inf')
    index_max = float('-inf')

    # First pass to find t_max and index_max
    for node in graph.node_indexes():
        time = graph.nodes()[node]['time']
        index = graph.nodes()[node]['index']
        if time is not None:
            t_max = max(t_max, time)
        index_max = max(index_max, index)

    # Second pass for plotting nodes
    for node in graph.node_indexes():
        is_boundary = graph.nodes()[node]['is_boundary']
        time = graph.nodes()[node]['time']
        index = graph.nodes()[node]['index']

        if time is None:
            time = t_max / 2
            index = -0.5 if index == 0 else index_max + 0.5

        node_info_dict[node] = {
            'is_boundary': is_boundary, 'time': time, 'index': index}
        color = 'r' if is_boundary else 'b'

        ax.scatter(time, index, c=color)

    # ... (rest of the code remains the same up to the edge plotting section)

    # Plot edges
    for i, edge in enumerate(graph.edge_list()):
        src, tgt = edge
        edge_data = graph.edges()[i]
        weight = edge_data.get('weight', 1)
        qubits = edge_data['qubits']

        src_time = node_info_dict[src]['time']
        tgt_time = node_info_dict[tgt]['time']
        src_index = node_info_dict[src]['index']
        tgt_index = node_info_dict[tgt]['index']

        edge_color = 'm'
        edge_label = f"{qubits} w:{weight}"

        if not qubits:  # Time edges
            edge_color = 'green'
            edge_label = f"w:{weight}"

        if src_time != tgt_time and src_index != tgt_index:  # Mixed edges
            edge_color = 'grey'  # 'm' stands for magenta, change as you like
            edge_label = f"{qubits} w:{weight}"

        ax.plot([src_time, tgt_time], [src_index, tgt_index],
                color=edge_color, linestyle='--')

        mid_time = (src_time + tgt_time) / 2
        ax.text(mid_time, (src_index + tgt_index) / 2, edge_label)

    # Add legend in the top right corner, ensuring no overlap
    ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Boundary Qubits', markersize=10, markerfacecolor='red'),
                       plt.Line2D([0], [0], marker='o', color='w',
                                  label='Check Nodes', markersize=10, markerfacecolor='blue'),
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
