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

    std::vector<std::pair<int64_t, int64_t>> decode_to_edges_array(UserGraph &self, const std::vector<uint64_t> &detection_events) {
        auto &mwpm = self.get_mwpm_with_search_graph();
        auto edges = std::make_unique<std::vector<int64_t>>();
        edges->reserve(detection_events.size() / 2);
        pm::decode_detection_events_to_edges(mwpm, detection_events, *edges);

        std::vector<std::pair<int64_t, int64_t>> edge_pairs;
        edge_pairs.reserve(edges->size() / 2);

        for (size_t i = 0; i < edges->size(); i += 2) {
            edge_pairs.emplace_back((*edges)[i], (*edges)[i + 1]);
        }

        return edge_pairs;
    }


    UserGraph detector_error_model_to_user_graph_private(const stim::DetectorErrorModel& detector_error_model) {
        UserGraph user_graph(detector_error_model.count_detectors(), detector_error_model.count_observables());
        iter_detector_error_model_edges(
            detector_error_model, [&](double p, const std::vector<size_t>& detectors, std::vector<size_t>& observables) {
                user_graph.handle_dem_instruction(p, detectors, observables);
            });
        return user_graph;
    }

    void append_custom(
        stim::Circuit &self,
        const std::string &gate_name,
        const std::vector<uint32_t> &targets,
        const std::vector<double> &args) {

            stim::GateType gate_id = stringToGateType(gate_name);
            std::vector<stim::GateTarget> converted;
            converted.reserve(targets.size());
            for (auto e : targets) {
                converted.push_back({e});
            }

            self.safe_append(gate_id, converted, args);
        }

    void circuit_append(
        stim::Circuit &self,
        const std::string &gate_name,
        const std::vector<uint32_t> &targets,
        const std::vector<double> &args,
        bool backwards_compat) {    

        if (backwards_compat && stim::GATE_DATA.at(gate_name).arg_count == 1) {
            // If the gate expects exactly one argument but none provided, use a default value.
            if (args.empty()) {
                self.safe_append_ua(gate_name, targets, 0.0);
            } else if (args.size() == 1) {
                self.safe_append_ua(gate_name, targets, args[0]);
            } else {
                throw std::invalid_argument("Gate expects exactly one argument.");
            }
        } else {
            // For gates that do not specifically require backwards compatibility handling or
            // can accept multiple arguments.
            if (args.size() == 1) {
                // If only one argument is provided, you could still use safe_append_ua for consistency.
                self.safe_append_ua(gate_name, targets, args[0]);
            } else {
                // Use safe_append_u for gates with multiple arguments or no arguments at all.
                self.safe_append_u(gate_name, targets, args);
            }
        }
    }

    stim::Circuit modify_circuit(const stim::Circuit& og_circuit) {
        stim::Circuit new_circuit;

        // Assuming you have "circuit.stim" file in the current directory.
        const char* filename = "circuit.stim";
        FILE* file = fopen(filename, "r"); // Open the file for reading.
        if (!file) {
            throw std::runtime_error("Failed to open file: " + std::string(filename));
        }

        // Append operations from the file to the new circuit.
        new_circuit.append_from_file(file, true);

        // Don't forget to close the file.
        fclose(file);

        return new_circuit;
    }


    // stim::Circuit modify_circuit(const stim::Circuit& og_circuit) {
    //     stim::Circuit new_circuit;
        
    //     for (const auto& instr : og_circuit.operations) {
    //         if (instr.gate_type == stim::GateType::M || instr.gate_type == stim::GateType::MR) {
    //             std::vector<uint32_t> converted_targets;
    //             for (const auto& target : instr.targets) {
    //                 new_circuit.safe_append_u((instr.gate_type == stim::GateType::M) ? "M" : "MR", {target.data}, {0.01*target.data});
    //             }
    //         } else if (instr.gate_type == stim::GateType::REPEAT) {
    //             stim::Circuit repeat_block;
    //             const stim::Circuit& block_body = instr.repeat_block_body(og_circuit);
    //             for (const auto& repeat_instr : block_body.operations) {
    //                 if (repeat_instr.gate_type == stim::GateType::M || repeat_instr.gate_type == stim::GateType::MR) {
    //                     for (const auto& target : repeat_instr.targets) {
    //                         repeat_block.safe_append_u((repeat_instr.gate_type == stim::GateType::M) ? "M" : "MR", {target.data}, {0.01*target.data});
    //                     }
    //                 } else {
    //                     // For all other types of instructions, append them with the same arguments as in the original instruction.
    //                     std::vector<double> args(repeat_instr.args.begin(), repeat_instr.args.end());
    //                     std::vector<uint32_t> converted_targets;
    //                     for (const auto& target : repeat_instr.targets) {
    //                         converted_targets.push_back(target.data);
    //                     }
    //                     std::string gate_str = gateTypeToString(repeat_instr.gate_type);
    //                     std::cout << "Gate string: " << gate_str << std::endl;
    //                     repeat_block.safe_append_u(gate_str, converted_targets, args);
    //                     // repeat_block.safe_append(repeat_instr.gate_type, repeat_instr.targets, repeat_instr.args);
    //                 }
    //             }
    //             // Append the repeat block to the new circuit.
    //             new_circuit.append_repeat_block(instr.repeat_block_rep_count(), repeat_block);       
    //         } else {
    //             // For all other types of instructions, append them with the same arguments as in the original instruction.
    //             std::vector<double> args(instr.args.begin(), instr.args.end());
    //             std::vector<uint32_t> converted_targets;
    //             for (const auto& target : instr.targets) {
    //                 converted_targets.push_back(target.data);
    //             }                
    //             new_circuit.safe_append_u(gateTypeToString(instr.gate_type), converted_targets, args);
    //         }
    //     }

    //     return new_circuit;
    // }

    // stim::Circuit modify_circuit(const stim::Circuit& og_circuit) {
    //     stim::Circuit new_circuit;
        
    //     for (const auto& instr : og_circuit.operations) {
    //         if (instr.gate_type == stim::GateType::M || instr.gate_type == stim::GateType::MR) {
    //             std::vector<uint32_t> converted_targets;
    //             for (const auto& target : instr.targets) {
    //                 new_circuit.safe_append_u((instr.gate_type == stim::GateType::M) ? "M" : "MR", {target.data}, {0.01*target.data});
    //             }
    //         } else if (instr.gate_type == stim::GateType::REPEAT) {
    //             const stim::Circuit& block_body = instr.repeat_block_body(og_circuit);
    //             for (size_t rep=0; rep < instr.repeat_block_rep_count(); ++rep) {
    //                 for (const auto& repeat_instr : block_body.operations) {
    //                     if (repeat_instr.gate_type == stim::GateType::M || repeat_instr.gate_type == stim::GateType::MR) {
    //                         for (const auto& target : repeat_instr.targets) {
    //                             new_circuit.safe_append_u((repeat_instr.gate_type == stim::GateType::M) ? "M" : "MR", {target.data}, {0.01*target.data});
    //                         }
    //                     } else if (instr.gate_type == stim::GateType::PAULI_CHANNEL_1) {
    //                         throw std::invalid_argument("Pauli channel 1 not supported.");
    //                     } else if (instr.gate_type == stim::GateType::DEPOLARIZE2) {
    //                         std::vector<double> args(repeat_instr.args.begin(), repeat_instr.args.end());
    //                         std::vector<uint32_t> converted_targets;
    //                         for (const auto& target : repeat_instr.targets) {
    //                             converted_targets.push_back(target.data);
    //                         }
    //                         throw std::invalid_argument("target len: " + std::to_string(converted_targets.size()));
    //                         new_circuit.safe_append_u("DEPOLARIZE2", converted_targets, args);
    //                     } else {
    //                         // For all other types of instructions, append them with the same arguments as in the original instruction.
    //                         std::vector<double> args(repeat_instr.args.begin(), repeat_instr.args.end());
    //                         std::vector<uint32_t> converted_targets;
    //                         for (const auto& target : repeat_instr.targets) {
    //                             converted_targets.push_back(target.data);
    //                         }
    //                         // new_circuit.safe_append_u(gateTypeToString(repeat_instr.gate_type), converted_targets, args);
    //                         new_circuit.safe_append(instr.gate_type, instr.targets, instr.args);
    //                     }
    //                 }
    //             }     
    //         } else if (instr.gate_type == stim::GateType::PAULI_CHANNEL_1) {
    //             throw std::invalid_argument("Pauli channel 1 not supported.");
    //         } else if (instr.gate_type == stim::GateType::DEPOLARIZE2) {
    //             std::vector<double> args(instr.args.begin(), instr.args.end());
    //             std::vector<uint32_t> converted_targets;
    //             for (const auto& target : instr.targets) {
    //                 converted_targets.push_back(target.data);
    //             }

    //             throw std::invalid_argument("target len: " + std::to_string(converted_targets.size()));
    //             std::cout << "DEPOLARIZE2" << std::endl;
    //             std::cout << "Targets len: " << converted_targets.size() << std::endl;

    //             new_circuit.safe_append_u("DEPOLARIZE2", converted_targets, args);
    //         } else {
    //             // For all other types of instructions, append them with the same arguments as in the original instruction.
    //             std::vector<double> args(instr.args.begin(), instr.args.end());
    //             std::vector<uint32_t> converted_targets;
    //             for (const auto& target : instr.targets) {
    //                 converted_targets.push_back(target.data);
    //             }                
    //             // new_circuit.safe_append_u(gateTypeToString(instr.gate_type), converted_targets, args);
    //             new_circuit.safe_append(instr.gate_type, instr.targets, instr.args);
    //         }
    //     }

    //     return new_circuit;
    // }


    stim::DetectorErrorModel createDetectorErrorModel(const stim::Circuit& circuit,
                                            bool decompose_errors,
                                            bool flatten_loops,
                                            bool allow_gauge_detectors,
                                            double approximate_disjoint_errors,
                                            bool ignore_decomposition_failures,
                                            bool block_decomposition_from_introducing_remnant_edges) {
    // Call the ErrorAnalyzer::circuit_to_detector_error_model function
    return stim::ErrorAnalyzer::circuit_to_detector_error_model(
                circuit,
                decompose_errors,
                !flatten_loops,
                allow_gauge_detectors,
                approximate_disjoint_errors,
                ignore_decomposition_failures,
                block_decomposition_from_introducing_remnant_edges);
}




}


std::vector<int> counts_to_det_syndr(const std::string& input_str, bool _resets, bool verbose, bool reverse) {
    // Step 1: Reverse the input string
    std::string reversed_str;
    if (reverse) {
        std::string reversed(input_str.rbegin(), input_str.rend());
        reversed_str = reversed;
    } else {
        reversed_str = input_str;
    }

    if (verbose) {std::cout << "Reversed string: " << reversed_str << std::endl;} ////////////////
    // if (verbose) {std::cout << "WARNING: not reversing the string for speed reasons!" << std::endl;} ////////////////

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


std::map<std::pair<int, int>, ErrorProbabilities> calculate_naive_error_probs(
    const pm::UserGraph& graph, 
    const std::map<std::string, size_t>& counts,
    bool _resets) {
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

    std::map<std::pair<int, int>, ErrorProbabilities> error_probs;
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

std::map<std::pair<int, int>, ErrorProbabilities> calculate_spitz_error_probs(
    const pm::UserGraph& graph, 
    const std::map<std::string, size_t>& counts,
    bool _resets) {

    // Initialize variables
    std::map<int, double> av_v, prod;
    std::map<std::pair<int, int>, double> av_vv, av_xor;
    std::map<int, std::vector<int>> neighbours;
    std::map<std::pair<int, int>, ErrorProbabilities> error_probs;
    std::set<int> boundary;

    size_t total_shots = 0;
    for (const auto& pair : counts) {
        total_shots += pair.second;  // Accumulate the number of total_shots
    }

    // Initialize neighbors and average values
    for (const auto& edge : graph.edges) {
        if (edge.node1 == SIZE_MAX || edge.node1 == -1 || edge.node2 == SIZE_MAX || edge.node2 == -1) {
            boundary.insert(edge.node1 == SIZE_MAX || edge.node1 == -1 ? edge.node2 : edge.node1);
            continue;
        }
        av_v[edge.node1] = 0;
        av_v[edge.node2] = 0;
        av_vv[{edge.node1, edge.node2}] = 0;
        av_xor[{edge.node1, edge.node2}] = 0;
        neighbours[edge.node1].push_back(edge.node2);
        neighbours[edge.node2].push_back(edge.node1);
    }

    // Process each error string and update counts
    for (const auto& pair : counts) {
        const auto& error_string = pair.first;
        const auto count = pair.second;
        auto error_nodes = counts_to_det_syndr(error_string, _resets); // Convert error string to a list of error nodes

        // Convert the error vector to a list of checked node indices
        std::vector<int> checked_node_indices;
        for (size_t i = 0; i < error_nodes.size(); ++i) {
            if (error_nodes[i] == 1) {
                checked_node_indices.push_back(i);
            }
        }

        // Update av_v, a_vv, av_xor for each checked node
        for (int node_idx : checked_node_indices) {
            av_v[node_idx] += count;  // Node at this index is in error

            // Iterate over each neighbor of the current node
            for (int neighbor_idx : neighbours[node_idx]) {
                if (std::find(checked_node_indices.begin(), checked_node_indices.end(), neighbor_idx) != checked_node_indices.end()) {
                    // Both the current node and its neighbor are in error
                    av_vv[{node_idx, neighbor_idx}] += count;
                } else {
                    // Only the current node is in error, not its neighbor
                    av_xor[{node_idx, neighbor_idx}] += count;
                }
            }
        }
    }

    // Normalize the averages
    for (auto& kv : av_v) {
        kv.second /= static_cast<double>(total_shots);
    }
    for (auto& kv : av_vv) {
        kv.second /= static_cast<double>(total_shots);
    }
    for (auto& kv : av_xor) {
        kv.second /= static_cast<double>(total_shots);
    }

    // Step 1: Process each edge and calculate error probabilities
    std::vector<std::pair<int, int>> boundary_edges;
    for (const auto& edge : graph.edges) {
        int node1 = edge.node1, node2 = edge.node2;
        for (const auto& edge : graph.edges) {
            if (node1 == -1 || node2 == -1) {
                // Correctly pushing a pair of integers into boundary_edges
                boundary_edges.push_back(std::make_pair(node1, node2));
                continue;
            }

            // Calculate error probability for regular edges
            if (1 - 2 * av_xor[{node1, node2}] != 0) {
                double x = (av_vv[{node1, node2}] - av_v[node1] * av_v[node2]) / (1 - 2 * av_xor[{node1, node2}]);
                if (x < 0.25) {
                    error_probs[{node1, node2}] = {std::max(0.0, 0.5 - std::sqrt(0.25 - x)), 0};
                } else {
                    error_probs[{node1, node2}] = {std::nan(""), 0};
                }
            } else {
                error_probs[{node1, node2}] = {std::nan(""), 0};
            }
        }
    }

    // Step 2: Process boundary edges and update prod
    for (const auto& boundary_edge : boundary_edges) {
        int boundary_node = (boundary_edge.first == -1) ? boundary_edge.second : boundary_edge.first;
        prod[boundary_node] = 1.0;

        for (const auto& neighbor_idx : neighbours[boundary_node]) {
            if (error_probs.find({boundary_node, neighbor_idx}) != error_probs.end()) {
                prod[boundary_node] *= 1 - 2 * error_probs[{boundary_node, neighbor_idx}].probability;
            } else if (error_probs.find({neighbor_idx, boundary_node}) != error_probs.end()) {
                prod[boundary_node] *= 1 - 2 * error_probs[{neighbor_idx, boundary_node}].probability;
            }
        }
    }

    // Step 3: Assign error probabilities to boundary edges
    for (const auto& boundary_edge : boundary_edges) {
        int boundary_node = (boundary_edge.first == -1) ? boundary_edge.second : boundary_edge.first;
        error_probs[boundary_edge] = {0.5 + (av_v[boundary_node] - 0.5) / prod[boundary_node], 0};
    }

    return error_probs;
}


std::string gateTypeToString(stim::GateType gate_type) {
    switch (gate_type) {
        case stim::GateType::NOT_A_GATE:
            return "NOT_A_GATE";
        case stim::GateType::DETECTOR:
            return "DETECTOR";
        case stim::GateType::OBSERVABLE_INCLUDE:
            return "OBSERVABLE_INCLUDE";
        case stim::GateType::TICK:
            return "TICK";
        case stim::GateType::QUBIT_COORDS:
            return "QUBIT_COORDS";
        case stim::GateType::SHIFT_COORDS:
            return "SHIFT_COORDS";
        case stim::GateType::REPEAT:
            return "REPEAT";
        case stim::GateType::MPAD:
            return "MPAD";
        case stim::GateType::MX:
            return "MX";
        case stim::GateType::MY:
            return "MY";
        case stim::GateType::M:
            return "M";
        case stim::GateType::MRX:
            return "MRX";
        case stim::GateType::MRY:
            return "MRY";
        case stim::GateType::MR:
            return "MR";
        case stim::GateType::RX:
            return "RX";
        case stim::GateType::RY:
            return "RY";
        case stim::GateType::R:
            return "R";
        case stim::GateType::MPP:
            return "MPP";
        case stim::GateType::XCX:
            return "XCX";
        case stim::GateType::XCY:
            return "XCY";
        case stim::GateType::XCZ:
            return "XCZ";
        case stim::GateType::YCX:
            return "YCX";
        case stim::GateType::YCY:
            return "YCY";
        case stim::GateType::YCZ:
            return "YCZ";
        case stim::GateType::CX:
            return "CX";
        case stim::GateType::CY:
            return "CY";
        case stim::GateType::CZ:
            return "CZ";
        case stim::GateType::H:
            return "H";
        case stim::GateType::H_XY:
            return "H_XY";
        case stim::GateType::H_YZ:
            return "H_YZ";
        case stim::GateType::DEPOLARIZE1:
            return "DEPOLARIZE1";
        case stim::GateType::DEPOLARIZE2:
            return "DEPOLARIZE2";
        case stim::GateType::X_ERROR:
            return "X_ERROR";
        case stim::GateType::Y_ERROR:
            return "Y_ERROR";
        case stim::GateType::Z_ERROR:
            return "Z_ERROR";
        case stim::GateType::PAULI_CHANNEL_1:
            return "PAULI_CHANNEL_1";
        case stim::GateType::PAULI_CHANNEL_2:
            return "PAULI_CHANNEL_2";
        case stim::GateType::E:
            return "E";
        case stim::GateType::ELSE_CORRELATED_ERROR:
            return "ELSE_CORRELATED_ERROR";
        case stim::GateType::HERALDED_ERASE:
            return "HERALDED_ERASE";
        case stim::GateType::HERALDED_PAULI_CHANNEL_1:
            return "HERALDED_PAULI_CHANNEL_1";
        case stim::GateType::I:
            return "I";
        case stim::GateType::X:
            return "X";
        case stim::GateType::Y:
            return "Y";
        case stim::GateType::Z:
            return "Z";
        case stim::GateType::C_XYZ:
            return "C_XYZ";
        case stim::GateType::C_ZYX:
            return "C_ZYX";
        case stim::GateType::SQRT_X:
            return "SQRT_X";
        case stim::GateType::SQRT_X_DAG:
            return "SQRT_X_DAG";
        case stim::GateType::SQRT_Y:
            return "SQRT_Y";
        case stim::GateType::SQRT_Y_DAG:
            return "SQRT_Y_DAG";
        case stim::GateType::S:
            return "S";
        case stim::GateType::S_DAG:
            return "S_DAG";
        case stim::GateType::SQRT_XX:
            return "SQRT_XX";
        case stim::GateType::SQRT_XX_DAG:
            return "SQRT_XX_DAG";
        case stim::GateType::SQRT_YY:
            return "SQRT_YY";
        case stim::GateType::SQRT_YY_DAG:
            return "SQRT_YY_DAG";
        case stim::GateType::SQRT_ZZ:
            return "SQRT_ZZ";
        case stim::GateType::SQRT_ZZ_DAG:
            return "SQRT_ZZ_DAG";
        case stim::GateType::SWAP:
            return "SWAP";
        case stim::GateType::ISWAP:
            return "ISWAP";
        case stim::GateType::CXSWAP:
            return "CXSWAP";
        case stim::GateType::SWAPCX:
            return "SWAPCX";
        case stim::GateType::ISWAP_DAG:
            return "ISWAP_DAG";
        case stim::GateType::MXX:
            return "MXX";
        case stim::GateType::MYY:
            return "MYY";
        case stim::GateType::MZZ:
            return "MZZ";
        default:
            return "UNKNOWN_GATE"; // Return this if the gate type is not recognized
    }
}


stim::GateType stringToGateType(const std::string &gate_string) {
    if (gate_string == "NOT_A_GATE") {
        return stim::GateType::NOT_A_GATE;
    } else if (gate_string == "DETECTOR") {
        return stim::GateType::DETECTOR;
    } else if (gate_string == "OBSERVABLE_INCLUDE") {
        return stim::GateType::OBSERVABLE_INCLUDE;
    } else if (gate_string == "TICK") {
        return stim::GateType::TICK;
    } else if (gate_string == "QUBIT_COORDS") {
        return stim::GateType::QUBIT_COORDS;
    } else if (gate_string == "SHIFT_COORDS") {
        return stim::GateType::SHIFT_COORDS;
    } else if (gate_string == "REPEAT") {
        return stim::GateType::REPEAT;
    } else if (gate_string == "MPAD") {
        return stim::GateType::MPAD;
    } else if (gate_string == "MX") {
        return stim::GateType::MX;
    } else if (gate_string == "MY") {
        return stim::GateType::MY;
    } else if (gate_string == "M") {
        return stim::GateType::M;
    } else if (gate_string == "MRX") {
        return stim::GateType::MRX;
    } else if (gate_string == "MRY") {
        return stim::GateType::MRY;
    } else if (gate_string == "MR") {
        return stim::GateType::MR;
    } else if (gate_string == "RX") {
        return stim::GateType::RX;
    } else if (gate_string == "RY") {
        return stim::GateType::RY;
    } else if (gate_string == "R") {
        return stim::GateType::R;
    } else if (gate_string == "MPP") {
        return stim::GateType::MPP;
    } else if (gate_string == "XCX") {
        return stim::GateType::XCX;
    } else if (gate_string == "XCY") {
        return stim::GateType::XCY;
    } else if (gate_string == "XCZ") {
        return stim::GateType::XCZ;
    } else if (gate_string == "YCX") {
        return stim::GateType::YCX;
    } else if (gate_string == "YCY") {
        return stim::GateType::YCY;
    } else if (gate_string == "YCZ") {
        return stim::GateType::YCZ;
    } else if (gate_string == "CX") {
        return stim::GateType::CX;
    } else if (gate_string == "CY") {
        return stim::GateType::CY;
    } else if (gate_string == "CZ") {
        return stim::GateType::CZ;
    } else if (gate_string == "H") {
        return stim::GateType::H;
    } else if (gate_string == "H_XY") {
        return stim::GateType::H_XY;
    } else if (gate_string == "H_YZ") {
        return stim::GateType::H_YZ;
    } else if (gate_string == "DEPOLARIZE1") {
        return stim::GateType::DEPOLARIZE1;
    } else if (gate_string == "DEPOLARIZE2") {
        return stim::GateType::DEPOLARIZE2;
    } else if (gate_string == "X_ERROR") {
        return stim::GateType::X_ERROR;
    } else if (gate_string == "Y_ERROR") {
        return stim::GateType::Y_ERROR;
    } else if (gate_string == "Z_ERROR") {
        return stim::GateType::Z_ERROR;
    } else if (gate_string == "PAULI_CHANNEL_1") {
        return stim::GateType::PAULI_CHANNEL_1;
    } else if (gate_string == "PAULI_CHANNEL_2") {
        return stim::GateType::PAULI_CHANNEL_2;
    } else if (gate_string == "E") {
        return stim::GateType::E;
    } else if (gate_string == "ELSE_CORRELATED_ERROR") {
        return stim::GateType::ELSE_CORRELATED_ERROR;
    } else if (gate_string == "HERALDED_ERASE") {
        return stim::GateType::HERALDED_ERASE;
    } else if (gate_string == "HERALDED_PAULI_CHANNEL_1") {
        return stim::GateType::HERALDED_PAULI_CHANNEL_1;
    } else if (gate_string == "I") {
        return stim::GateType::I;
    } else if (gate_string == "X") {
        return stim::GateType::X;
    } else if (gate_string == "Y") {
        return stim::GateType::Y;
    } else if (gate_string == "Z") {
        return stim::GateType::Z;
    } else if (gate_string == "C_XYZ") {
        return stim::GateType::C_XYZ;
    } else if (gate_string == "C_ZYX") {
        return stim::GateType::C_ZYX;
    } else if (gate_string == "SQRT_X") {
        return stim::GateType::SQRT_X;
    } else if (gate_string == "SQRT_X_DAG") {
        return stim::GateType::SQRT_X_DAG;
    } else if (gate_string == "SQRT_Y") {
        return stim::GateType::SQRT_Y;
    } else if (gate_string == "SQRT_Y_DAG") {
        return stim::GateType::SQRT_Y_DAG;
    } else if (gate_string == "S") {
        return stim::GateType::S;
    } else if (gate_string == "S_DAG") {
        return stim::GateType::S_DAG;
    } else if (gate_string == "SQRT_XX") {
        return stim::GateType::SQRT_XX;
    } else if (gate_string == "SQRT_XX_DAG") {
        return stim::GateType::SQRT_XX_DAG;
    } else if (gate_string == "SQRT_YY") {
        return stim::GateType::SQRT_YY;
    } else if (gate_string == "SQRT_YY_DAG") {
        return stim::GateType::SQRT_YY_DAG;
    } else if (gate_string == "SQRT_ZZ") {
        return stim::GateType::SQRT_ZZ;
    } else if (gate_string == "SQRT_ZZ_DAG") {
        return stim::GateType::SQRT_ZZ_DAG;
    } else if (gate_string == "SWAP") {
        return stim::GateType::SWAP;
    } else if (gate_string == "ISWAP") {
        return stim::GateType::ISWAP;
    } else if (gate_string == "CXSWAP") {
        return stim::GateType::CXSWAP;
    } else if (gate_string == "SWAPCX") {
        return stim::GateType::SWAPCX;
    } else if (gate_string == "ISWAP_DAG") {
        return stim::GateType::ISWAP_DAG;
    } else if (gate_string == "MXX") {
        return stim::GateType::MXX;
    } else if (gate_string == "MYY") {
        return stim::GateType::MYY;
    } else if (gate_string == "MZZ") {
        return stim::GateType::MZZ;
    } 
}




// stim::Circuit modify_circuit(const stim::Circuit& og_circuit) {
//     stim::Circuit new_circuit;
    
//     for (const auto& instr : og_circuit.operations) {
//         if (instr.gate_type == stim::GateType::M || instr.gate_type == stim::GateType::MR) {
//             std::vector<uint32_t> converted_targets;
//             for (const auto& target : instr.targets) {
//                 new_circuit.safe_append_u((instr.gate_type == stim::GateType::M) ? "M" : "MR", {target.data}, {0.4*target.data});
//             }
//         } else if (instr.gate_type == stim::GateType::REPEAT) {
//             stim::Circuit repeat_block;
//             const stim::Circuit& block_body = instr.repeat_block_body(og_circuit);
//             for (const auto& repeat_instr : block_body.operations) {
//                 if (repeat_instr.gate_type == stim::GateType::M || repeat_instr.gate_type == stim::GateType::MR) {
//                     for (const auto& target : repeat_instr.targets) {
//                         repeat_block.safe_append_u((repeat_instr.gate_type == stim::GateType::M) ? "M" : "MR", {target.data}, {0.01*target.data});
//                     }
//                 } else {
//                     // For all other types of instructions, append them with the same arguments as in the original instruction.
//                     std::vector<double> args(repeat_instr.args.begin(), repeat_instr.args.end());
//                     std::vector<uint32_t> converted_targets;
//                     for (const auto& target : repeat_instr.targets) {
//                         converted_targets.push_back(target.data);
//                     }
//                     repeat_block.safe_append_u(gateTypeToString(repeat_instr.gate_type), converted_targets, args);
//                 }
//             }
//             // Append the repeat block to the new circuit.
//             new_circuit.append_repeat_block(instr.repeat_block_rep_count(), repeat_block);       
//         } else {
//             // For all other types of instructions, append them with the same arguments as in the original instruction.
//             std::vector<double> args(instr.args.begin(), instr.args.end());
//             std::vector<uint32_t> converted_targets;
//             for (const auto& target : instr.targets) {
//                 converted_targets.push_back(target.data);
//             }                
//             new_circuit.safe_append_u(gateTypeToString(instr.gate_type), converted_targets, args);
//         }
//     }

//     return new_circuit;
// }