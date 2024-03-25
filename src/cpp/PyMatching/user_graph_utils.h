#ifndef USER_GRAPH_UTILS_H
#define USER_GRAPH_UTILS_H

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

struct ErrorProbabilities {
    double probability;
    size_t samples;
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

    std::vector<std::pair<int64_t, int64_t>> decode_to_edges_array(UserGraph &self, const std::vector<uint64_t> &detection_events);

    UserGraph detector_error_model_to_user_graph_private(const stim::DetectorErrorModel& detector_error_model);

    void append_custom(
        stim::Circuit &self,
        const std::string &gate_name,
        const std::vector<uint32_t> &targets,
        const std::vector<double> &args);

    void circuit_append(
        stim::Circuit &self,
        const std::string &gate_name,
        const std::vector<uint32_t> &targets,
        const std::vector<double> &args,
        bool backwards_compat = true);

    stim::Circuit modify_circuit(const stim::Circuit& og_circuit);

    stim::DetectorErrorModel createDetectorErrorModel(const stim::Circuit& circuit,
                                            bool decompose_errors = false,
                                            bool flatten_loops = false,
                                            bool allow_gauge_detectors = false,
                                            double approximate_disjoint_errors = false,
                                            bool ignore_decomposition_failures = false,
                                            bool block_decomposition_from_introducing_remnant_edges = false);
}

std::string gateTypeToString(stim::GateType gate_type);

stim::GateType stringToGateType(const std::string &gate_string);

std::vector<int> counts_to_det_syndr(const std::string& input_str, bool _resets = false, bool verbose = false, bool reverse = true);

std::vector<uint64_t> syndromeArrayToDetectionEvents(const std::vector<int>& z, int num_detectors, int boundary_length);

std::map<std::pair<int, int>, ErrorProbabilities> calculate_naive_error_probs(
    const pm::UserGraph& graph, 
    const std::map<std::string, size_t>& counts,
    bool _resets);

std::map<std::pair<int, int>, ErrorProbabilities> calculate_spitz_error_probs(
    const pm::UserGraph& graph, 
    const std::map<std::string, size_t>& counts,
    bool _resets);


#endif // USER_GRAPH_UTILS_H
