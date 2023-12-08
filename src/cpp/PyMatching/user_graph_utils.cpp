#include "user_graph_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"



namespace pm {

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


    std::pair<std::vector<uint8_t>, double> decode(UserGraph &self, const std::vector<uint64_t> &detection_events) {
        auto &mwpm = self.get_mwpm();
        auto obs_crossed = std::make_unique<std::vector<uint8_t>>(self.get_num_observables(), 0);
        pm::total_weight_int weight = 0;
        pm::decode_detection_events(mwpm, detection_events, obs_crossed->data(), weight);
        double rescaled_weight = static_cast<double>(weight) / mwpm.flooder.graph.normalising_constant;
        std::vector<uint8_t> obs_crossed_vec(obs_crossed->begin(), obs_crossed->end());
        return {obs_crossed_vec, rescaled_weight};
    }



}


std::vector<int> counts_to_det_syndr(const std::string& input_str, bool _resets, bool verbose) {
    // Step 1: Reverse the input string
    std::string reversed_str(input_str.rbegin(), input_str.rend());

    if (verbose) {std::cout << "Reversed string: " << reversed_str << std::endl;} ////////////////

    // Step 2: Separate the count string
    size_t space_pos = reversed_str.rfind(" ");
    std::string count_str = reversed_str.substr(space_pos + 1);
    std::string check_str = reversed_str.substr(0, space_pos);

    // Splitting check_str into parts
    std::vector<std::string> check_str_parts;
    size_t start = 0;
    space_pos = check_str.find(" ", start);
    while (space_pos != std::string::npos) {
        check_str_parts.push_back(check_str.substr(start, space_pos - start));
        start = space_pos + 1;
        space_pos = check_str.find(" ", start);
    }
    check_str_parts.push_back(check_str.substr(start));

    // The number of measurement rounds T
    int T = check_str_parts.size();


    if (verbose) {std::cout << "Count string: " << count_str << std::endl; ////////////////
                std::cout << "Check string: " << check_str << std::endl; ////////////////
                std::cout << "Number of measurement rounds (T):" << T << std::endl;} ////////////////


    // Check if the count string is one bit longer than the following strings
    for (const std::string& part : check_str_parts) {
        if (count_str.length() != part.length() + 1) {
            throw std::invalid_argument("Count string must be one bit longer than each check string.");
        }
    }

    // Step 3: Transpose the check_str into T parts
    std::vector<std::string> check_str_parts_T(T);
    for (int i = 0; i < T; ++i) {
        for (size_t j = 0; j < check_str_parts.size(); ++j) {
            if (i < check_str_parts[j].length()) {
                check_str_parts_T[i] += check_str_parts[j][i];
            }
        }
    }


    if (verbose) {std::cout << "Check string parts T (not T):" ;
    for (int i = 0; i < T; ++i) {
        std::cout  << check_str_parts[i] << std::endl; ////////////////
    }}

    // Step 4: Initialize detector string list with T+1 empty strings
    std::vector<std::string> detector_str_parts(T + 1);

    // Step 5: Set the first part of the detector string
    detector_str_parts[0] = check_str_parts[0];

    std::vector<std::string> meas_str_parts(T + 1);
    meas_str_parts[0] = check_str_parts[0];

    if (!_resets) {
        //Step 5.5 No resets -> compute the syndrome measurement outcomes
        for (int i = 1; i < T; ++i) {
            for (size_t j = 0; j < check_str_parts[i].length(); ++j) {
                int bit1 = check_str_parts[i - 1][j] - '0';
                int bit2 = check_str_parts[i][j] - '0';
                meas_str_parts[i] += std::to_string((bit1 + bit2) % 2);
            }
        }
        
        if (verbose) {std::cout << "_reset = False: Syndrome meas outcomes:" ;
        for (int i = 0; i < T; ++i) {
            std::cout  << meas_str_parts[i] << std::endl; ////////////////
        }}

        // Compute parts 2 to T of the detector string
        for (int i = 1; i < T; ++i) {
            for (size_t j = 0; j < meas_str_parts[i].length(); ++j) {
                int bit1 = meas_str_parts[i - 1][j] - '0';
                int bit2 = meas_str_parts[i][j] - '0';
                detector_str_parts[i] += std::to_string((bit1 + bit2) % 2);
            }
        }
        
        if (verbose) {std::cout << "_reset = False: Detector str parts:" ;
        for (int i = 0; i < T; ++i) {
            std::cout  << detector_str_parts[i] << std::endl; ////////////////
        }}

    } else {
        // For _resets = true, directly use check_str_parts_T for detector string parts
        for (int i = 1; i < T; ++i) {
            for (size_t j = 0; j < check_str_parts[i].length(); ++j) {
                int bit1 = check_str_parts[i - 1][j] - '0';
                int bit2 = check_str_parts[i][j] - '0';
                detector_str_parts[i] += std::to_string((bit1 + bit2) % 2);
            }
        }

        if (verbose) {std::cout << "_reset = True: Detector str parts:" ;
        for (int i = 0; i < T; ++i) {
            std::cout  << detector_str_parts[i]<< std::endl; ////////////////
        }}
    }

    // Step 7: Compute the XOR string from the count string
    std::string xor_result;
    for (size_t i = 0; i < count_str.length() - 1; ++i) {
        int bit1 = count_str[i] - '0';
        int bit2 = count_str[i + 1] - '0';
        xor_result += std::to_string((bit1 + bit2) % 2);
    }

    if (verbose) {std::cout << "XOR result: " << xor_result << std::endl;} ////////////////


    if (!_resets && verbose){
        std::cout << "_reset = False: meas_str_parts[T-1]:" << meas_str_parts[T-1] << std::endl; ////////////////
    }
    // Compute the (T+1)th part of the detector string using the T-1 part of the check str
    std::string& last_part = (!_resets) ? meas_str_parts[T - 1] : check_str_parts[T - 1];
    for (size_t i = 0; i < xor_result.length(); ++i) {
        int bit1 = xor_result[i] - '0';
        int bit2 = (i < last_part.length()) ? (last_part[i] - '0') : 0;  // Ensure handling if lengths differ
        detector_str_parts[T] += std::to_string((bit1 + bit2) % 2);
    }


    if (verbose) {std::cout << "Detector str parts after count detector:" ; 
    for (int i = 0; i < T + 1; ++i) {
        std::cout << detector_str_parts[i]; ////////////////
    }}

    // Convert detector string parts to a single vector of ints
    std::vector<int> numpy_list;
    for (const auto& part : detector_str_parts) {
        for (char bit : part) {
            numpy_list.push_back(bit - '0');
        }
    }

    return numpy_list;
}


std::vector<uint64_t> syndromeArrayToDetectionEvents(const std::vector<int>& z, int num_detectors, int boundary_length) {
    
    // num_detectors throug UserGraph.get_num_detectors() 
    // boundary_length through UserGraph.get_boundary().size()
    // Check if the input vector is empty
    if (z.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }

    std::vector<uint64_t> detection_events;

    // Handling 1D array case with the specified condition
    if (num_detectors <= z.size() && z.size() <= num_detectors + boundary_length) {
        for (size_t i = 0; i < z.size(); ++i) {
            if (z[i] != 0) {
                detection_events.push_back(i);
            }
        }
    } else {
        throw std::invalid_argument("Invalid size of the syndrome vector");
    }

    // TODO: Handling 2D array case (Hardcoded for RepCodes)

    return detection_events;
}


/////////////////// get_error_probs ///////////////////////


std::map<std::pair<int, int>, std::optional<ErrorProbabilities>> calculate_naive_error_probs(
    const pm::UserGraph& graph, 
    const std::map<std::string, size_t>& counts,
    bool _resets
) {
    // Map to store count for each combination ("00", "01", "10", "11") for each edge
    std::map<std::pair<size_t, size_t>, std::map<std::string, size_t>> count;

    // Initialize count for each edge
    for (const auto& edge : graph.edges) {
        count[{edge.node1, edge.node2}] = {{"00", 0}, {"01", 0}, {"10", 0}, {"11", 0}};
    }

    // Process each error string and update counts
    for (const auto& pair : counts) {
        const auto& error_string = pair.first;
        const auto error_nodes = counts_to_det_syndr(error_string, _resets);

        for (const auto& edge : graph.edges) {
            std::string element;
            if (edge.node1 == SIZE_MAX) {
                // If node2 is in error, both elements are "1", otherwise both are "0"
                element = error_nodes[edge.node2] ? "11" : "00";
            } else if (edge.node2 == SIZE_MAX) {
                // If node1 is in error, both elements are "1", otherwise both are "0"
                element = error_nodes[edge.node1] ? "11" : "00";
            } else {
                element += error_nodes[edge.node1] ? "1" : "0"; // Check error status for node1
                element += error_nodes[edge.node2] ? "1" : "0"; // Check error status for node2
            }

            count[{edge.node1, edge.node2}][element] += pair.second;
        }
    }
    
    std::map<std::pair<int, int>, std::optional<ErrorProbabilities>> error_probs;
    for (const auto& item : count) {
        const auto& edge = item.first;
        const auto& histogram = item.second;

        double ratio = (histogram.at("00") > 0) ? static_cast<double>(histogram.at("11")) / histogram.at("00") : std::numeric_limits<double>::quiet_NaN();
        if (edge.first == SIZE_MAX || edge.second == SIZE_MAX) {
            ratio /= 2.0;
        }
        double p = ratio / (1.0 + ratio);

        error_probs[{static_cast<int>(edge.first), static_cast<int>(edge.second)}] = ErrorProbabilities{p, histogram.at("11") + histogram.at("00")};
    }

    return error_probs;
}