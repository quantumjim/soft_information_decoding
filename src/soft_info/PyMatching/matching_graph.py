# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-11-01

import pymatching
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
            p_mixed = p_data  #TODO find a better ratio 
            #p_mixed = 1e-10
            new_weight = -np.log(p_mixed / (1 - p_mixed))
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


def draw_matching_graph(matching, d, T, syndromes=None, matched_edges=None, figsize=(8, 6)):
    
    matched_edges = matched_edges.tolist() if matched_edges is not None else None
    G = nx.Graph()
    pos = {}
    edge_colors = []
    edge_widths = []
    node_colors = []

    # Define normal and highlighted edge widths
    normal_edge_width = 1
    highlighted_edge_width = 5

    # Add all nodes to the graph with their positions and initial colors
    for i in range((d-1)*(T+1)):
        x_pos = i % (d-1)
        y_pos = i // (d-1)
        G.add_node(i)  # Explicitly add the node
        pos[i] = (x_pos, -y_pos)
        if syndromes is not None and syndromes[i] == 1:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')
    
    # Add edges to the graph and keep track of their attributes for drawing
    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        if tgt_node is not None:
            G.add_edge(src_node, tgt_node, weight=edge_data['weight'])
            # Now we don't need to append the color and widths here, since we'll draw them individually

    
    # Draw the graph
    plt.figure(figsize=figsize)
    
    nx.draw(G, pos, labels={node: node for node in G.nodes()}, with_labels=True, 
            node_color=node_colors,
            font_weight='bold', node_size=500, font_size=12)
    
    # Draw the graph edges individually with their specific colors and widths
    for edge in G.edges():
        src_node, tgt_node = edge
        #sorted_edge = tuple(sorted([src_node, tgt_node]))
        if matched_edges is not None and ([src_node, tgt_node] in matched_edges or [tgt_node, src_node] in matched_edges):
            color = 'blue'
            width = highlighted_edge_width
        else:
            color = 'black'
            width = normal_edge_width
        
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=width, edge_color=color)


    # Draw edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_labels = {edge: f"{weight:.2f}" for edge, weight in edge_weights.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    for edge in matching.edges():
        src_node, tgt_node, edge_data = edge
        if tgt_node is None:
            x_src = src_node % (d-1)
            y_src = src_node // (d-1)
            if matched_edges is not None:
                if [src_node, -1] in matched_edges or [-1, src_node] in matched_edges:
                    color = 'orange' if edge_data.get('fault_ids') else 'blue'
                    lw = highlighted_edge_width
                else:
                    color = 'r' if edge_data.get('fault_ids') else 'k'
                    lw = normal_edge_width
            else:
                    color = 'r' if edge_data.get('fault_ids') else 'k'
                    lw = normal_edge_width
            weight_text = f"{edge_data['weight']:.2f}" if 'weight' in edge_data else ""
            if x_src == 0:
                plt.plot([x_src, x_src - 0.5], [-y_src, -y_src], color=color, lw=lw)
                plt.text(x_src - 0.45, -y_src + 0.03, weight_text, fontsize=10)
            elif x_src == d - 2:
                plt.plot([x_src, x_src + 0.5], [-y_src, -y_src], color=color, lw=lw)
                plt.text(x_src + 0.2, -y_src + 0.03, weight_text, fontsize=10)
    
    # Create legend handles
    red_node = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                             markersize=10, label='Syndromes')
    blue_edge = mlines.Line2D([], [], color='blue', lw=2, label='Matched edges')
    orange_edge = mlines.Line2D([], [], color='orange', lw=2, label='Matched LOGICAL edges')

    # Create a legend
    legend_handles = []
    if syndromes is not None:
        legend_handles.append(red_node)
    if matched_edges is not None:
        legend_handles.append(blue_edge)
        legend_handles.append(orange_edge)
    

    # Locate the legend
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)

    plt.axis('scaled')
    plt.subplots_adjust(right=0.7)
    plt.show()