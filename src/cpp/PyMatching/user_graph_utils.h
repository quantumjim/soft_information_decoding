#include <set>
#include <vector>
#include <utility> // For std::pair
#include <string>
#include <unordered_map>
#include <utility> 
#include "pymatching/sparse_blossom/driver/user_graph.h" // Include necessary headers for declarations

struct EdgeAttributes {
    std::set<size_t> fault_ids;
    double weight;
    double error_probability;
};

struct EdgeProperties {
    size_t node1;
    size_t node2;
    EdgeAttributes attributes;
};

namespace pm {

    MERGE_STRATEGY merge_strategy_from_string(const std::string &merge_strategy);

    void add_edge(UserGraph &graph, int64_t node1, int64_t node2, 
                           const std::set<size_t> &observables, double weight, 
                           double error_probability, const std::string &merge_strategy);

    void add_boundary_edge(UserGraph &graph, int64_t node, 
                           const std::set<size_t> &observables, double weight, 
                           double error_probability, const std::string &merge_strategy);

    std::vector<EdgeProperties> get_edges(const pm::UserGraph& graph);

    std::pair<std::vector<uint8_t>, double> decode(UserGraph &self, const std::vector<uint64_t> &detection_events);
}

std::vector<int> counts_to_det_syndr(const std::string& input_str, bool _resets = false, bool verbose = false);

std::vector<uint64_t> syndromeArrayToDetectionEvents(const std::vector<int>& z, int num_detectors, int boundary_length);
