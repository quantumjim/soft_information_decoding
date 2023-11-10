import pymatching
import numpy as np

from src.soft_info.PyMatching.matching_graph import reweight_edges_to_one

def test_reweight_edges_to_one():
    # Create a pymatching.Matching object with some edges
    matching = pymatching.Matching()
    num_edges = 5
    for i in range(num_edges):
        matching.add_edge(i, i+1, weight=np.random.random(), error_probability=np.random.random())

    # Apply the reweight_edges_to_one function
    reweight_edges_to_one(matching)

    # Assert that all edge weights are now 1
    for edge in matching.edges():
        _, _, edge_data = edge
        assert edge_data['weight'] == 1, "Edge weight is not equal to 1"
