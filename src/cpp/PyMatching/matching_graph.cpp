#include "matching_graph.h"
#include "user_graph_utils.h"
#include "pymatching/sparse_blossom/driver/user_graph.h" // Include necessary headers for declarations
#include <iostream>

#include <vector>
#include <set>
#include <map>
#include <cmath>

namespace pm {
    void soft_reweight_pymatching(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict, 
        float p_data, float p_mixed, float common_measure) {

        // Set default values
        p_data = (p_data != -1) ? p_data : 6.836e-3; // Sherbrooke median
        p_mixed = (p_mixed != -1) ? p_mixed : 0;

        // Distance
        int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

        // Get edges
        std::vector<EdgeProperties> edges = pm::get_edges(matching);

        for (const auto& edge : edges) { 
            bool _has_time_component = false;
            int src_node = edge.node1;
            int tgt_node = edge.node2;
            auto& edge_data = edge.attributes;

            double new_weight; // Initialize new weight

            if (tgt_node == -1) {
                // Boundary edge
                new_weight = -std::log(p_data / (1 - p_data));
                if (common_measure != -1) {
                    new_weight = std::round(new_weight / common_measure) * common_measure;
                }
                pm::add_boundary_edge(matching, src_node, edge_data.fault_ids, new_weight,
                                    edge_data.error_probability, "replace"); 
                continue;
            }

            if (tgt_node == src_node + 1){ // always first pos smaller TODO: check if that is correct
                // Data edge
                new_weight = -std::log(p_data / (1 - p_data));
                if (common_measure != -1) {
                    new_weight = std::round(new_weight / common_measure) * common_measure;
                }
                pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                            edge_data.error_probability, "replace"); 
                continue;
            }

            if (tgt_node == src_node + (distance-1) + 1) { // Hardcoded for RepCodes
                // Time-Data edge
                // double p_mixed = p_data / 50;
                new_weight = -std::log(p_mixed / (1 - p_mixed));
                if (common_measure != -1) {
                    new_weight = std::round(new_weight / common_measure) * common_measure;
                }
                pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                            edge_data.error_probability, "replace"); 
                continue;
            }

            if (tgt_node == src_node + (distance-1)) { // Hardcoded for RepCodes
                // Time edge
                // TODO: implement adding a new edge with p_meas
                new_weight = 0;
                _has_time_component = true;
            }

            if (_has_time_component) { // weird structure but could be useful in the future for mixed edges         
                // Structure of IQ data = [link_0, link_1, link_3, link_0, link_1, .., code_qubit_1, ...]
                // equivalent to       = [node_0, node_1, node_3, node_4, node_5, .. ]
                // =>

                // Step 1: Select the correct qubit_idx
                int qubit_idx = qubit_mapping.at(src_node);
                const auto& grid_data = kde_grid_dict.at(qubit_idx);

                // Step 2: Rescale the corresponding IQ point
                std::complex<double> iq_point = not_scaled_IQ_data(0, src_node);
                const auto& [real_params, imag_params] = scaler_params_dict.at(qubit_idx);
                double real_scaled = (std::real(iq_point) - real_params.first) / real_params.second;
                double imag_scaled = (std::imag(iq_point) - imag_params.first) / imag_params.second;
                Eigen::Vector2d scaled_point = {real_scaled, imag_scaled};

                // Step 3: Get the llh_ratio
                double llh_weight = llh_ratio(scaled_point, grid_data);
                new_weight += llh_weight;

                if (common_measure != -1) {
                    new_weight = std::round(new_weight / common_measure) * common_measure;
                }

                // Update the edge weight
                pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                            edge_data.error_probability, "replace");
            }
        }
    }

    void reweight_edges_to_one(UserGraph &matching) {
        // Get edges
        std::vector<EdgeProperties> edges = pm::get_edges(matching);

        for (const auto& edge : edges) { 
            bool _has_time_component = false;
            int src_node = edge.node1;
            int tgt_node = edge.node2;
            auto& edge_data = edge.attributes;

            double new_weight = 1.0; 

            if (tgt_node == -1) {
                // Boundary edge
                pm::add_boundary_edge(matching, src_node, edge_data.fault_ids, new_weight,
                                    edge_data.error_probability, "replace"); 
                continue;
            }else{
                // Data/Time/Mixed edge
                pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                            edge_data.error_probability, "replace"); 
                continue;
            }
        }
    }

    int decode_IQ_shots(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict, 
        float p_data, float p_mixed, float common_measure) {
        
        int numErrors = 0;
        for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
            Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
            auto counts = get_counts(not_scaled_IQ_shot_matrix, qubit_mapping, kde_grid_dict, scaler_params_dict, synd_rounds);
            std::string count_key = counts.begin()->first;

            // add copying the graph to recompute weights to 1 or something 
            soft_reweight_pymatching(matching, not_scaled_IQ_shot_matrix, synd_rounds, qubit_mapping, kde_grid_dict, scaler_params_dict, p_data, p_mixed, common_measure);

            auto det_syndromes = counts_to_det_syndr(count_key, false, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());

            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

            int actual_observable = (static_cast<int>(count_key[0]) - '0') % 2;  // Convert first character to int and modulo 2
            // Check if predicted_observables is not empty and compare the first element
            if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
                numErrors++;  // Increment error count if they don't match
            }

        }

        return numErrors;
    }

    int decode_IQ_shots_flat(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict) {
        int numErrors = 0;
        reweight_edges_to_one(matching);
        for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
            Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
            auto counts = get_counts(not_scaled_IQ_shot_matrix, qubit_mapping, kde_grid_dict, scaler_params_dict, synd_rounds);
            std::string count_key = counts.begin()->first;

            auto det_syndromes = counts_to_det_syndr(count_key, false, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());

            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

            int actual_observable = (static_cast<int>(count_key[0]) - '0') % 2;  // Convert first character to int and modulo 2
            // Check if predicted_observables is not empty and compare the first element
            if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
                numErrors++;  // Increment error count if they don't match
            }

        }

        return numErrors;
    }

}

void processGraph_test(pm::UserGraph& graph) {
    std::cout << "Processing graph" << std::endl;
    // Implement the logic for processing the graph
}


// Comand to compile:
// cd src/cpp/Pymatching
// g++ matching_graph.cpp -o matching_graph \
// -I"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/src" \
// -I"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/Stim/src" \
// -L"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/bazel-bin" \
// -llibpymatching -std=c++20
