# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-11-01

import pymatching
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib.patches as mpatches

from ..Hardware import get_repcode_IQ_map
from ..UnionFind import llh_ratio

from ..Plotting.general import apply_formatting


#TODO add correct error rates on the edges
def soft_reweight_pymatching(matching: pymatching.Matching,  d: int, T: int, IQ_data,
                             kde_dict: dict, layout: list, scaler_dict: dict,
                             p_data: float = None, p_meas: float = None, common_measure=None,
                             verbose: bool = False):
    
    p_data = p_data if p_data is not None else 6.836e-3  # Sherbrooke median
    p_meas = p_meas if p_meas is not None else 0

    if layout is not None:
        qubit_mapping = get_repcode_IQ_map(layout, T) #Hardcoded for repetition code

    for edge in matching.edges():
        _has_time_component = False
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

            #_has_time_component = False
            continue


        elif tgt_node == src_node + 1:  # always first pos the smaller
            # Data edge
            new_weight = -np.log(p_data / (1 - p_data))
            #_has_time_component = False
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
            p_mixed = p_data/50  #TODO find a better ratio 
            #p_mixed = 1e-10
            new_weight = -np.log(p_mixed / (1 - p_mixed))
            #_has_time_component = False # JRW: Diag are like data errors
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

            
def bezier_midpoint_and_tangent(p0, p1, p2, t=0.65):
    """Calculate point and tangent vector at t on a quadratic Bezier curve."""
    # Ensure inputs are numpy arrays for element-wise operations
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    point = (1-t)**2 * p0 + 2 * (1-t) * t * p1 + t**2 * p2
    tangent = 2 * (1-t) * (p1 - p0) + 2 * t * (p2 - p1)
    return point, tangent

def draw_curved_edge(ax, pos, src_node, tgt_node, color, width, edge_weight, font_size, scale_factor):
    src_pos = pos[src_node]
    tgt_pos = pos[tgt_node]

    # Calculate midpoint and control point for Bezier curve
    mid_pos = np.mean([src_pos, tgt_pos], axis=0)
    control_point_offset = np.array([0.5, 2]) * scale_factor
    control_point = mid_pos + control_point_offset


    # Calculate midpoint and tangent for text positioning and orientation
    midpoint, tangent = bezier_midpoint_and_tangent(src_pos, control_point, tgt_pos)
    angle_rad = np.arctan2(tangent[1], tangent[0])
    angle_deg = np.degrees(angle_rad)

    # Draw Bezier curve
    path = mpatches.Path([src_pos, control_point, tgt_pos], [mpatches.Path.MOVETO, mpatches.Path.CURVE3, mpatches.Path.CURVE3])
    patch = mpatches.PathPatch(path, facecolor='none', lw=width, edgecolor=color)
    ax.add_patch(patch)

    # Draw text with rotation aligned with the tangent
    plt.text(midpoint[0], midpoint[1], f"{edge_weight:.2f}", color=color, fontsize=font_size,
             ha='center', va='center', rotation=angle_deg,
             bbox=dict(facecolor='white', edgecolor='none', pad=3*scale_factor, alpha=1))



def draw_matching_graph(matching=None, d=3, T=3, syndromes=None, matched_edges=None, figsize=(8, 6), scale_factor=1, edge_list=None, dpi=150):
    
    try:
        matched_edges = matched_edges.tolist() if matched_edges is not None else None
    except AttributeError:
        matched_edges = matched_edges if matched_edges is not None else None

    G = nx.Graph()
    pos = {}
    edge_colors = []
    edge_widths = []
    node_colors = []

    # Define normal and highlighted edge widths
    normal_edge_width = 1
    highlighted_edge_width = 5

    # Add all nodes to the graph with their positions afnd initial colors
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
    iterable = matching.edges() if edge_list is None else edge_list
    for edge in iterable:
        src_node, tgt_node, edge_data = edge if edge_list is None else (edge.node1, edge.node2, edge.attributes)
        edge_data = {'weight': edge_data.weight, "fault_ids": edge_data.fault_ids} if edge_list is not None else edge_data
        if tgt_node is not None and tgt_node != 18446744073709551615:
            G.add_edge(src_node, tgt_node, weight=edge_data['weight'])
            # Now we don't need to append the color and widths here, since we'll draw them individually

    # Adjust sizes and fonts
    node_size = 100 * scale_factor  # Scale down node size
    font_size = 12 * scale_factor   # Scale down font size
    normal_edge_width *= scale_factor  # Scale down edge width
    highlighted_edge_width *= scale_factor

    # Scale node positions
    pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}

    apply_formatting(dpi=dpi)
    fig = plt.figure()
    # Draw the graph
    # if dpi is not None:
    #     plt.figure(figsize=figsize, dpi=dpi)
    # else:
    #     plt.figure(figsize=figsize)
    
    nx.draw(G, pos, labels={node: node for node in G.nodes()}, with_labels=True, 
            node_color=node_colors,
            font_weight='bold', node_size=node_size, font_size=font_size)
    
    # Draw the graph edges individually with their specific colors and widths

    edge_weights = nx.get_edge_attributes(G, 'weight')
    ax = plt.gca()
    for edge in G.edges():
        src_node, tgt_node = edge
        edge_weight = edge_weights.get(edge, 0)
        if tgt_node == src_node + 2 * (d - 1):  # Check for NNN condition
        # Determine the color and width based on whether the edge is in matched_edges
            if matched_edges is not None and (edge in matched_edges or (edge[1], edge[0]) in matched_edges):
                color = 'blue'  # Color for matched edges
                width = highlighted_edge_width*0.5
                ntnn_fs = font_size
            else:
                color = 'black'  # Default color for NNN edges
                width = normal_edge_width*0.5
                ntnn_fs = font_size*0.8

            # Draw an arc for NNN edges
            draw_curved_edge(ax, pos, src_node, tgt_node, color, width, edge_weight, ntnn_fs, scale_factor)
            continue

        #sorted_edge = tuple(sorted([src_node, tgt_node]))
        if matched_edges is not None and (([src_node, tgt_node] in matched_edges or [tgt_node, src_node] in matched_edges)
                                           or ((src_node, tgt_node) in matched_edges or (tgt_node, src_node) in matched_edges)):
            color = 'blue'
            width = highlighted_edge_width
        else:
            color = 'black'
            width = normal_edge_width
        
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=width, edge_color=color)


    # Draw edge weights
    edge_labels = {}
    for edge in G.edges():
        src_node, tgt_node = edge
        if tgt_node != src_node + 2 * (d - 1):  # Exclude NNN edges
            weight = edge_weights.get(edge, 0)
            edge_labels[edge] = f"{weight:.2f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size)

    for edge in iterable:
        src_node, tgt_node, edge_data = edge if edge_list is None else (edge.node1, edge.node2, edge.attributes)
        edge_data = {'weight': edge_data.weight, "fault_ids": edge_data.fault_ids} if edge_list is not None else edge_data
        if tgt_node is None or tgt_node == 18446744073709551615:
            x_src = src_node % (d-1)
            y_src = src_node // (d-1)
            x_src_old = x_src
            x_src *= scale_factor
            y_src *= scale_factor
            if matched_edges is not None:
                if ([src_node, -1] in matched_edges or [-1, src_node] in matched_edges 
                    or (src_node, -1) in matched_edges or (-1, src_node) in matched_edges):
                    color = 'orange' if edge_data.get('fault_ids') else 'blue'
                    lw = highlighted_edge_width
                else:
                    color = 'r' if edge_data.get('fault_ids') else 'k'
                    lw = normal_edge_width
            else:
                    color = 'r' if edge_data.get('fault_ids') else 'k'
                    lw = normal_edge_width
            weight_text = f"{edge_data['weight']:.2f}" if 'weight' in edge_data else ""
            if x_src_old == 0:
                plt.plot([x_src, x_src - 0.5*scale_factor], [-y_src, -y_src], color=color, lw=lw)
                plt.text(x_src - 0.35*scale_factor, -y_src - 0.015*scale_factor, weight_text, fontsize=font_size,
                         bbox=dict(facecolor='white', edgecolor='none', pad=3*scale_factor, alpha=1))
            elif x_src_old == d - 2:
                plt.plot([x_src, x_src + 0.5*scale_factor], [-y_src, -y_src], color=color, lw=lw)
                plt.text(x_src + 0.2*scale_factor, -y_src - 0.015*scale_factor, weight_text, fontsize=font_size,
                         bbox=dict(facecolor='white', edgecolor='none', pad=3*scale_factor, alpha=1))


    
    # Create legend handles
    red_node = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                             markersize=10*scale_factor, label='Syndromes')
    blue_edge = mlines.Line2D([], [], color='blue', lw=2*scale_factor, label='Matched edges')
    orange_edge = mlines.Line2D([], [], color='orange', lw=2*scale_factor, label='Matched LOGICAL edges')

    # Create a legend
    legend_handles = []
    if syndromes is not None:
        legend_handles.append(red_node)
    if matched_edges is not None:
        legend_handles.append(blue_edge)
        legend_handles.append(orange_edge)
    

    # Locate the legend
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0, fontsize=font_size)


    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)

    plt.axis('scaled')
    plt.subplots_adjust(right=0.7)
    plt.show()
    return fig