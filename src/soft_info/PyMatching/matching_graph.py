# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-11-01

import pymatching
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from ..Hardware import get_repcode_IQ_map
from ..UnionFind import llh_ratio


#TODO add correct error rates on the edges
def soft_reweight_pymatching(matching: pymatching.Matching,  d: int, T: int, IQ_data,
                             kde_dict: dict, layout: list, scaler_dict: dict,
                             p_data: float = None, p_meas: float = None, common_measure=None,
                             verbose: bool = False):

    p_data = p_data if p_data is not None else 6.836e-3  # Sherbrooke median
    p_meas = p_meas if p_meas is not None else 0

    if layout is not None:
        qubit_mapping = get_repcode_IQ_map(layout, T)

    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        if verbose:
            print("\nEdge:", (src_node, tgt_node))
        fault_ids = edge_data.get('fault_ids', set())
        error_probability = edge_data.get('error_probability', -1.0)

        if tgt_node is None:  # always second pose None
            # Boundary edge (logical on it)
            new_weight = -np.log(p_data / (1 - p_data))

            if common_measure is not None:
                new_weight = round(
                    new_weight / common_measure) * common_measure

            matching.add_boundary_edge(src_node, weight=new_weight, fault_ids=fault_ids,
                                       error_probability=error_probability, merge_strategy="replace")
            if verbose:
                print("Boundary edge weight: ", new_weight)

            _has_time_component = False
            continue

        elif tgt_node == src_node + 1:  # always first pos the smaller
            # Data edge
            new_weight = -np.log(p_data / (1 - p_data))
            if common_measure is not None:
                new_weight = round(
                    new_weight / common_measure) * common_measure
            if verbose:
                print("Data edge weight: ", new_weight)

        elif tgt_node == src_node + (d-1):
            # Time edge
            # TODO implement adding a new edge for hard meas flip
            new_weight = 0  # -np.log(p_meas / (1 - p_meas))
            _has_time_component = True
            if verbose:
                print("Time edge weight: ", new_weight)

        elif tgt_node == src_node + (d-1) + 1:
            # mixed edge
            new_weight = -np.log(p_data / (1 - p_data))
            _has_time_component = False # JRW: Diag are like data errors
            if common_measure is not None:
                new_weight = round(
                    new_weight / common_measure) * common_measure
            if verbose:
                print("Mixed edge weight: ", new_weight)

        if _has_time_component:
            # Structure of IQ data = [link_0, link_1, link_3, link_0, link_1, .., code_qubit_1, ...]
            # equivalent to       = [node_0, node_1, node_3, node_4, node_5, .. ]
            # =>
            IQ_point = IQ_data[src_node]
            layout_qubit_idx = qubit_mapping[src_node]
            kde_0, kde_1 = kde_dict.get(layout_qubit_idx, (None, None))
            scaler = scaler_dict.get(layout_qubit_idx, None)
            llh_weight = llh_ratio(IQ_point, kde_0, kde_1, scaler)

            if verbose:
                print("LLH weight: ", llh_weight)

            new_weight += llh_weight

            # Round the weights to common measure
            if common_measure is not None:
                new_weight = round(
                    new_weight / common_measure) * common_measure

        # Update the edge weight
        matching.add_edge(src_node, tgt_node, weight=new_weight, fault_ids=fault_ids,
                          error_probability=error_probability, merge_strategy="replace")


def reweight_edges_to_one(matching: pymatching.Matching):
    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        fault_ids = edge_data.get('fault_ids', set())
        error_probability = edge_data.get('error_probability', -1.0)

        if tgt_node is None:
            matching.add_boundary_edge(src_node, weight=1, fault_ids=fault_ids,
                                       error_probability=error_probability, merge_strategy="replace")
        else:
            matching.add_edge(src_node, tgt_node, weight=1, fault_ids=fault_ids,
                              error_probability=error_probability, merge_strategy="replace")


def draw_matching_graph(matching, d, T):
    G = nx.Graph()
    pos = {}
    edge_colors = []

    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        if tgt_node is not None:
            G.add_edge(src_node, tgt_node, weight=edge_data['weight'])
            if edge_data.get('fault_ids'):
                edge_colors.append('r')
            else:
                edge_colors.append('k')

        x_src = src_node % (d-1)
        y_src = src_node // (d-1)
        pos[src_node] = (x_src, -y_src)

    nx.draw(G, pos, with_labels=True, node_color='white',
            edge_color=edge_colors, font_weight='bold', node_size=700, font_size=18)

    edge_weights = nx.get_edge_attributes(G, 'weight')
    labels = {k: f"{v:.2f}" for k, v in edge_weights.items()}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        if tgt_node is None:
            x_src = src_node % (d-1)
            y_src = src_node // (d-1)
            color = 'r' if edge_data.get('fault_ids') == set() else 'k'
            weight_text = f"{edge_data.get('weight'):.2f}"
            if x_src == 0:
                plt.plot([x_src, x_src - 0.5], [-y_src, -y_src], color=color)
                plt.text(x_src - 0.3, -y_src + 0.05, weight_text)
            elif x_src == d - 2:
                plt.plot([x_src, x_src + 0.5], [-y_src, -y_src], color=color)
                plt.text(x_src + 0.2, -y_src + 0.05, weight_text)

    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)

    plt.show()
