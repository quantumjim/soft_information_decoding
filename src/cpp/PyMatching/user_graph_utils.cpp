#include "user_graph_utils.h"

std::vector<EdgeProperties> get_edges(const pm::UserGraph& graph) {
    std::vector<EdgeProperties> edges;

    for (auto &e : graph.edges) { // Adjust based on how edges are stored in pm::UserGraph
        EdgeProperties edgeProps;
        edgeProps.node1 = e.node1;
        edgeProps.node2 = (e.node2 == SIZE_MAX) ? SIZE_MAX : e.node2; // SIZE_MAX to represent 'None' in Python

        EdgeAttributes attrs;
        attrs.error_probability = (e.error_probability < 0 || e.error_probability > 1) ? -1.0 : e.error_probability;
        attrs.fault_ids = std::set<size_t>(e.observable_indices.begin(), e.observable_indices.end());
        attrs.weight = e.weight;

        edgeProps.attributes = attrs;
        edges.push_back(edgeProps);
    }

    return edges;
}


namespace pm {
    MERGE_STRATEGY merge_strategy_from_string(const std::string &merge_strategy) {
        static const std::unordered_map<std::string, MERGE_STRATEGY> table = {
            {"disallow", MERGE_STRATEGY::DISALLOW},
            {"independent", MERGE_STRATEGY::INDEPENDENT},
            {"smallest-weight", MERGE_STRATEGY::SMALLEST_WEIGHT},
            {"keep-original", MERGE_STRATEGY::KEEP_ORIGINAL},
            {"replace", MERGE_STRATEGY::REPLACE}
        };

        auto it = table.find(merge_strategy);
        if (it != table.end()) {
            return it->second;
        } else {
            throw std::invalid_argument("Merge strategy \"" + merge_strategy + "\" not recognised.");
        }
    }

    void add_edge(UserGraph &graph, int64_t node1, int64_t node2, 
                           const std::set<size_t> &observables, double weight, 
                           double error_probability, const std::string &merge_strategy) {
        if (node1 < 0 || node2 < 0) {
            throw std::invalid_argument("Node indices must be non-negative.");
        }

        if (std::abs(weight) > MAX_USER_EDGE_WEIGHT) {
            // Handle the warning or error as appropriate
            return;
        }

        std::vector<size_t> observables_vec(observables.begin(), observables.end());
        graph.add_or_merge_edge(node1, node2, observables_vec, weight, 
                                error_probability, merge_strategy_from_string(merge_strategy));
    }

    void add_boundary_edge(UserGraph &graph, int64_t node, 
                           const std::set<size_t> &observables, double weight, 
                           double error_probability, const std::string &merge_strategy) {
        if (node < 0) {
            throw std::invalid_argument("Node index must be non-negative.");
        }

        if (std::abs(weight) > MAX_USER_EDGE_WEIGHT) {
            // Handle the warning or error as appropriate
            return;
        }

        std::vector<size_t> observables_vec(observables.begin(), observables.end());
        graph.add_or_merge_boundary_edge(node, observables_vec, weight, 
                                         error_probability, merge_strategy_from_string(merge_strategy));
    }

}

