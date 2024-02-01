#include "matching_graph.h"
#include "pymatching/sparse_blossom/driver/user_graph.h" // Include necessary headers for declarations
#include <iostream>

#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <stdexcept>

#include <chrono>
#include <omp.h>


namespace pm
{   
    /////// REWEIGHTING /////
    void soft_reweight_pymatching(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, GridData> &kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
        float p_data, float p_mixed, float common_measure,
        bool _adv_probs, bool _bimodal, const std::string &merge_strategy, float p_offset,
        float p_multiplicator, bool _ntnn_edges)
    {

        // Distance
        int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

        // Get edges
        std::vector<EdgeProperties> edges = pm::get_edges(matching);

        for (const auto &edge : edges)
        {
            bool _has_time_component = false;
            int src_node = edge.node1;
            int tgt_node = edge.node2;
            auto &edge_data = edge.attributes;

            double new_weight; // Initialize new weight

            if (tgt_node == -1)
            {
                // Boundary edge
                if (p_data != -1)
                {
                    new_weight = -std::log(p_data / (1 - p_data));
                    if (common_measure != -1)
                    {
                        new_weight = std::round(new_weight / common_measure) * common_measure;
                    }
                    pm::add_boundary_edge(matching, src_node, edge_data.fault_ids, new_weight,
                                          edge_data.error_probability, "replace");
                }
                continue;
            }

            if (tgt_node == src_node + 1)
            { // always first pos smaller TODO: check if that is correct
                // Data edge
                if (p_data != -1)
                {
                    new_weight = -std::log(p_data / (1 - p_data));
                    if (common_measure != -1)
                    {
                        new_weight = std::round(new_weight / common_measure) * common_measure;
                    }
                    pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                                 edge_data.error_probability, "replace");
                }
                continue;
            }

            if (tgt_node == src_node + (distance - 1) + 1)
            { // Hardcoded for RepCodes
                // Time-Data edge
                if (p_mixed != -1)
                {
                    new_weight = -std::log(p_mixed / (1 - p_mixed));
                    if (common_measure != -1)
                    {
                        new_weight = std::round(new_weight / common_measure) * common_measure;
                    }
                    pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                                 edge_data.error_probability, "replace");
                }
                continue;
            }

            if (tgt_node == src_node + (distance - 1))
            { // Hardcoded for RepCodes
                // Time edge
                // TODO: implement adding a new edge with p_meas
                new_weight = 0;
                _has_time_component = true;
            }

            if (_has_time_component)
            { // weird structure but could be useful in the future for mixed edges
                // Structure of IQ data = [link_0, link_1, link_3, link_0, link_1, .., code_qubit_1, ...]
                // equivalent to       = [node_0, node_1, node_3, node_4, node_5, .. ]
                // =>

                // Step 1: Select the correct qubit_idx
                int qubit_idx = qubit_mapping.at(src_node);
                const auto &grid_data = kde_grid_dict.at(qubit_idx);

                // Step 2: Rescale the corresponding IQ point
                std::complex<double> iq_point = not_scaled_IQ_data(0, src_node);
                const auto &[real_params, imag_params] = scaler_params_dict.at(qubit_idx);
                double real_scaled = (std::real(iq_point) - real_params.first) / real_params.second;
                double imag_scaled = (std::imag(iq_point) - imag_params.first) / imag_params.second;
                Eigen::Vector2d scaled_point = {real_scaled, imag_scaled};

                // Steb 2.2: get t-1 point
                Eigen::Vector2d scaled_point_tminus1 = {0, 0};
                if (_adv_probs and not _resets)
                {
                    if (src_node - (distance - 1) >= 0)
                    {
                        std::complex<double> iq_point_tminus1 = not_scaled_IQ_data(0, src_node - (distance - 1)); // Hardcoded for RepCodes
                        double real_scaled_tminus1 = (std::real(iq_point_tminus1) - real_params.first) / real_params.second;
                        double imag_scaled_tminus1 = (std::imag(iq_point_tminus1) - imag_params.first) / imag_params.second;
                        scaled_point_tminus1 = {real_scaled_tminus1, imag_scaled_tminus1};
                    }
                }

                // Step 3: Get the llh_ratio
                double llh_weight;
                double llh_weight_tminus1;
                if (_bimodal)
                {
                    // llh_weight = llh_ratio(scaled_point, grid_data, edge_data.error_probability);
                    llh_weight = llh_ratio(scaled_point, grid_data);
                    float llh_prob = 1 / (1 + (1 / std::exp(-llh_weight)));
                    // std::cout << "llh_prob before: " << llh_prob << std::endl;

                    llh_prob = llh_prob + edge_data.error_probability - 2 * llh_prob * edge_data.error_probability;
                    // added crossterms to not get negative weights
                    llh_weight = -std::log(llh_prob / (1 - llh_prob));
                    // std::cout << "llh_prob after: " << llh_prob << std::endl;
                    // std::cout << "llh_weight: " << llh_weight << std::endl;
                }
                else
                {
                    if (_adv_probs and not _resets)
                    {
                        llh_weight = llh_ratio(scaled_point, grid_data);
                        float p_soft_tminus1 = 0;
                        float p_soft = 1 / (1 + (1 / std::exp(-llh_weight)));
                        if (src_node - (distance - 1) >= 0)
                        {
                            llh_weight_tminus1 = llh_ratio(scaled_point_tminus1, grid_data);
                            p_soft_tminus1 = 1 / (1 + (1 / std::exp(-llh_weight_tminus1)));
                            if (_ntnn_edges) {
                                pm::add_edge(matching, src_node - (distance - 1), tgt_node, edge_data.fault_ids, // TODO: fix the fault ids to empty set
                                 llh_weight_tminus1, p_soft_tminus1, "replace");
                            }
                        }
                        float p_h = edge_data.error_probability * p_multiplicator;
                        // std::cout << "p_h: " << p_h << std::endl;
                        float edge_prob = p_h * (p_offset - p_soft_tminus1) * (p_offset - p_soft) + (1 - p_h) * p_soft_tminus1 * (p_offset - p_soft) + (1 - p_h) * (p_offset - p_soft_tminus1) * p_soft;   
                        if (_ntnn_edges){
                            if (tgt_node < (distance-1)*synd_rounds) {
                            edge_prob = p_h;
                            }
                            else {
                                edge_prob = p_h * (1-p_soft) + (1-p_h) * p_soft; // on last layer soft = hard flip => 2 indep mechanisms
                                // std::cout << "edge_prob: " << edge_prob << std::endl;
                            }
                        }
                        llh_weight = -std::log(edge_prob / (1 - edge_prob));
                    }
                    else
                    {
                        llh_weight = llh_ratio(scaled_point, grid_data);
                    }
                }
                new_weight += llh_weight;

                if (common_measure != -1)
                {
                    new_weight = std::round(new_weight / common_measure) * common_measure;
                }

                // Update the edge weight
                pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                             edge_data.error_probability, merge_strategy); // keeps the old error probability
            }
        }
    }

    void soft_reweight_1Dgauss(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, std::map<std::string, float>> &gauss_params_dict) {

            int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes
            std::vector<EdgeProperties> edges = pm::get_edges(matching);

            for (const auto &edge : edges) {
                int src_node = edge.node1;
                int tgt_node = edge.node2;
                auto &edge_data = edge.attributes;

                if (tgt_node == src_node + (distance - 1)) { // Time edges
                    int qubit_idx = qubit_mapping.at(src_node);
                    const auto &gauss_params = gauss_params_dict.at(qubit_idx);
                    std::complex<double> iq_point = not_scaled_IQ_data(0, src_node);    
                    double rpoint = std::real(iq_point);
                    
                    double rpoint_tminus1;
                    if (src_node - (distance - 1) >= 0) {
                        std::complex<double> iq_point_tminus1 = not_scaled_IQ_data(0, src_node - (distance - 1)); // Hardcoded for RepCodes
                        rpoint_tminus1 = std::real(iq_point_tminus1);
                    }

                    std::map<std::string, float> llh_params = llh_ratio_1Dgauss(rpoint, gauss_params);
                    std::map<std::string, float> llh_params_tminus1;

                    if (_resets) {
                        pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids,
                                    llh_params["weight"], llh_params["proba"], "independent");
                    } else {
                        if (src_node - (distance - 1) >= 0) {
                            llh_params_tminus1 = llh_ratio_1Dgauss(rpoint_tminus1, gauss_params);
                            pm::add_edge(matching, src_node - (distance - 1), tgt_node, 
                                        edge_data.fault_ids, llh_params_tminus1["weight"],
                                        llh_params_tminus1["proba"], "replace");
                        }
                    }  
                }
            }
    }

    
    // has cummulative reweighting problem + extremely slow (slower than python)
    void soft_reweight_kde(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        std::map<int, KDE_Result> kde_dict) {

            int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes
           
            for (int tgt_node=distance - 1; tgt_node < (distance-1)*synd_rounds; tgt_node++) {
                int src_node = tgt_node - (distance-1);
                int ntnn_src_node = tgt_node - 2*(distance-1);
                std::set<size_t> empty_set;

                if (_resets) {
                    std::complex<double> not_scaled_point = not_scaled_IQ_data(0, src_node); 
                    int qubit_idx = qubit_mapping.at(src_node);
                    auto &kde_entry = kde_dict.at(qubit_idx);                                    
                    std::map<std::string, float> llh_params = llh_ratio_kde(not_scaled_point, kde_entry);
                    pm::add_edge(matching, src_node, tgt_node, empty_set, llh_params["weight"], llh_params["proba"], "independent");
                } else {
                    if (ntnn_src_node >= 0) {
                        std::complex<double> not_scaled_point = not_scaled_IQ_data(0, ntnn_src_node); 
                        int qubit_idx = qubit_mapping.at(ntnn_src_node);
                        auto &kde_entry = kde_dict.at(ntnn_src_node);                                  
                        std::map<std::string, float> llh_params = llh_ratio_kde(not_scaled_point, kde_entry);
                        pm::add_edge(matching, ntnn_src_node, tgt_node, empty_set, llh_params["weight"], llh_params["proba"], "replace");
                    }
                }
            }
    }


    // void soft_reweight_kde(
    //     UserGraph &matching,
    //     const Eigen::MatrixXcd &not_scaled_IQ_data,
    //     int synd_rounds,
    //     bool _resets,
    //     const std::map<int, int> &qubit_mapping,
    //     std::map<int, KDE_Result> kde_dict) {

    //         int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes
    //         std::vector<EdgeProperties> edges = pm::get_edges(matching);

    //         for (const auto &edge : edges) {
    //             int src_node = edge.node1;
    //             int tgt_node = edge.node2;
    //             auto &edge_data = edge.attributes;

    //             if (tgt_node == src_node + (distance - 1)) { // Time edges
    //                 int qubit_idx = qubit_mapping.at(src_node);
    //                 auto &kde_entry = kde_dict.at(qubit_idx);
    //                 std::complex<double> not_scaled_point = not_scaled_IQ_data(0, src_node);   
    //                 std::complex<double> not_scaled_point_tminus1;
                    
    //                 if (src_node - (distance - 1) >= 0 and not _resets) {
    //                     std::complex<double> not_scaled_point_tminus1 = not_scaled_IQ_data(0, src_node - (distance - 1)); // Hardcoded for RepCodes
    //                 }

    //                 std::map<std::string, float> llh_params = llh_ratio_kde(not_scaled_point, kde_entry);
    //                 std::map<std::string, float> llh_params_tminus1;

    //                 if (_resets) {
    //                     pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids,
    //                                 llh_params["weight"], llh_params["proba"], "independent");
    //                 } else {
    //                     if (src_node - (distance - 1) >= 0) {
    //                         llh_params_tminus1 = llh_ratio_kde(not_scaled_point_tminus1, kde_entry);
    //                         pm::add_edge(matching, src_node - (distance - 1), tgt_node, 
    //                                     edge_data.fault_ids, llh_params_tminus1["weight"],
    //                                     llh_params_tminus1["proba"], "replace");
    //                     }
    //                 }  
    //             }
    //         }
    // }


    void reweight_edges_to_one(UserGraph &matching)
    {
        // Get edges
        std::vector<EdgeProperties> edges = pm::get_edges(matching);

        for (const auto &edge : edges)
        {
            bool _has_time_component = false;
            int src_node = edge.node1;
            int tgt_node = edge.node2;
            auto &edge_data = edge.attributes;

            float new_weight = 1.0;

            if (tgt_node == -1)
            {
                // Boundary edge
                pm::add_boundary_edge(matching, src_node, edge_data.fault_ids, new_weight,
                                      edge_data.error_probability, "replace");
                continue;
            }
            else
            {
                // Data/Time/Mixed edge
                pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                             edge_data.error_probability, "replace");
                continue;
            }
        }
    }

    void reweight_edges_informed(
        UserGraph &matching, float distance,
        float p_data, float p_mixed,
        float p_meas, float common_measure, bool _ntnn_edges)
    {
        // Get edges
        std::vector<EdgeProperties> edges = pm::get_edges(matching);

        for (const auto &edge : edges)
        {
            bool _has_time_component = false;
            int src_node = edge.node1;
            int tgt_node = edge.node2;
            auto &edge_data = edge.attributes;

            double new_weight;

            if (tgt_node == -1)
            {
                // Boundary edge
                if (p_data != -1)
                {
                    new_weight = -std::log(p_data / (1 - p_data));
                    if (common_measure != -1)
                    {
                        new_weight = std::round(new_weight / common_measure) * common_measure;
                    }
                    pm::add_boundary_edge(matching, src_node, edge_data.fault_ids, new_weight,
                                          edge_data.error_probability, "replace");
                }
                continue;
            }

            if (tgt_node == src_node + 1)
            { // always first pos smaller TODO: check if that is correct
                // Data edge
                if (p_data != -1)
                {
                    new_weight = -std::log(p_data / (1 - p_data));
                    if (common_measure != -1)
                    {
                        new_weight = std::round(new_weight / common_measure) * common_measure;
                    }
                    pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                                 edge_data.error_probability, "replace");
                }
                continue;
            }

            if (tgt_node == src_node + (distance - 1) + 1)
            { // Hardcoded for RepCodes
                // Mixed edge
                if (p_mixed != -1)
                {
                    new_weight = -std::log(p_mixed / (1 - p_mixed));
                    if (common_measure != -1)
                    {
                        new_weight = std::round(new_weight / common_measure) * common_measure;
                    }
                    pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                                 edge_data.error_probability, "replace");
                }
                continue;
            }

            if (tgt_node == src_node + (distance - 1))
            { // Hardcoded for RepCodes
                // Time edge
                if (src_node - (distance - 1) >= 0 and _ntnn_edges)
                    {   
                        pm::add_edge(matching, src_node - (distance - 1), tgt_node, edge_data.fault_ids, edge.attributes.weight,
                                     edge_data.error_probability, "replace");
                    }                                
                if (p_meas != -1)
                {
                    new_weight = -std::log(p_meas / (1 - p_meas));
                    if (common_measure != -1)
                    {
                        new_weight = std::round(new_weight / common_measure) * common_measure;
                    }
                    pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                                 edge_data.error_probability, "replace");
                    if (src_node - (distance - 1) >= 0 and _ntnn_edges)
                    {   
                        pm::add_edge(matching, src_node - (distance - 1), tgt_node, edge_data.fault_ids, new_weight,
                                     edge_data.error_probability, "replace");
                    }
                }
            }
        }
    }

    void reweight_edges_based_on_error_probs(UserGraph &matching, const std::map<std::string, size_t> &counts, bool _resets, const std::string &method)
    {

        std::map<std::pair<int, int>, ErrorProbabilities> error_probs;
        if (method == "naive")
        {
            error_probs = calculate_naive_error_probs(matching, counts, _resets);
        }
        else if (method == "spitz")
        {
            error_probs = calculate_spitz_error_probs(matching, counts, _resets);
        }
        else
        {
            throw std::invalid_argument("Invalid method: " + method);
        }

        // Get edges
        std::vector<EdgeProperties> edges = pm::get_edges(matching);

        for (const auto &edge : edges)
        {
            int src_node = edge.node1;
            int tgt_node = edge.node2;
            auto &edge_data = edge.attributes;

            // Check if this edge has an associated error probability
            auto it = error_probs.find({src_node, tgt_node});
            if (it != error_probs.end())
            { // Check if the edge is found in error_probs
                // Retrieve the error probability
                float error_probability = it->second.probability;

                // Compute the new weight as -log(p_data / (1 - p_data))
                float new_weight = (error_probability == 0 || error_probability == 1) ? std::numeric_limits<float>::infinity() : -std::log(error_probability / (1 - error_probability));

                if (tgt_node == -1)
                {
                    // Boundary edge
                    pm::add_boundary_edge(matching, src_node, edge_data.fault_ids, new_weight,
                                          error_probability, "replace");
                }
                else
                {
                    // Data/Time/Mixed edge
                    pm::add_edge(matching, src_node, tgt_node, edge_data.fault_ids, new_weight,
                                 error_probability, "replace");
                }
            }
        }
    }

    ////// DECODING ///////

    ShotErrorDetails createShotErrorDetails(
        UserGraph &matching,
        std::vector<uint64_t> &detectionEvents,
        std::vector<int> &det_syndromes)
    {

        ShotErrorDetails errorDetail;
        errorDetail.matched_edges = decode_to_edges_array(matching, detectionEvents);
        errorDetail.detection_syndromes = det_syndromes; // Assuming det_syndromes is available here
        errorDetail.edges = get_edges(matching);

        return errorDetail;
    }

    DetailedDecodeResult decode_IQ_shots(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, GridData> &kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
        float p_data, float p_mixed, float common_measure,
        bool _adv_probs, bool _bimodal, const std::string &merge_strategy, bool _detailed,
        float p_offset, float p_multiplicator, bool _ntnn_edges)
    {

        DetailedDecodeResult result;
        result.num_errors = 0;

        for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot)
        {
            Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
            auto counts = get_counts(not_scaled_IQ_shot_matrix, qubit_mapping, kde_grid_dict, scaler_params_dict, synd_rounds);
            std::string count_key = counts.begin()->first;

            // add copying the graph to recompute weights to 1 or something
            soft_reweight_pymatching(matching, not_scaled_IQ_shot_matrix, synd_rounds, _resets,
                                     qubit_mapping, kde_grid_dict, scaler_params_dict, p_data, p_mixed,
                                     common_measure, _adv_probs, _bimodal, merge_strategy, p_offset, p_multiplicator, _ntnn_edges);

            auto det_syndromes = counts_to_det_syndr(count_key, _resets, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());

            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

            int actual_observable = (static_cast<int>(count_key[0]) - logical) % 2; // Convert first character to int and modulo 2
            // int actual_observable = (static_cast<int>(count_key[0]) - '0') % 2;  // Convert first character to int and modulo 2
            // Check if predicted_observables is not empty and compare the first element
            if (_detailed)
            {
                ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                result.error_details.push_back(errorDetail);
            }
            if (!predicted_observables.empty() && predicted_observables[0] != actual_observable)
            {
                result.num_errors++; // Increment error count if they don't match
                result.indices.push_back(shot);
            }
        }
        return result;
    }

    DetailedDecodeResult decode_IQ_fast(
        stim::DetectorErrorModel detector_error_model,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, GridData> &kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
        bool _detailed,
        int nb_intervals) {
            
            DetailedDecodeResult result;
            result.num_errors = 0;
            result.error_details.resize(not_scaled_IQ_data.rows());

            std::set<size_t> empty_set;        
            int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes
            
            UserGraph matching;
            #pragma omp parallel private(matching)
            {
                matching = detector_error_model_to_user_graph_private(detector_error_model);
                #pragma omp for nowait  
                for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
                    Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
                    std::string count_key;
                    for (int msmt = 0; msmt < not_scaled_IQ_shot_matrix.cols(); ++msmt) { 
                        int qubit_idx = qubit_mapping.at(msmt);
                        std::complex<double> not_scaled_point = not_scaled_IQ_data(shot, msmt);
                        auto &grid_data = kde_grid_dict.at(qubit_idx);

                        const auto& [real_params, imag_params] = scaler_params_dict.at(qubit_idx);
                        double real_scaled = (std::real(not_scaled_point) - real_params.first) / real_params.second;
                        double imag_scaled = (std::imag(not_scaled_point) - imag_params.first) / imag_params.second;
                        Eigen::Vector2d scaled_point = {real_scaled, imag_scaled};

                        auto [outcome, density0, density1] = grid_lookup(scaled_point, grid_data);

                        double p_small;
                        double p_big;

                        if (outcome == 0) {
                            count_key += "0";    
                            p_small = std::exp(density1);
                            p_big = std::exp(density0);
                        } else {
                            count_key += "1";
                            p_small = std::exp(density0);
                            p_big = std::exp(density1);
                        }

                        if (msmt < (distance-1)*synd_rounds) {  
                            double p_soft = 1 / (1 + p_big/p_small);    

                            // round p_soft
                            if (nb_intervals != -1) {
                                double intervalSize = 0.5 / nb_intervals;
                                int intervalIndex = static_cast<int>(p_soft / intervalSize);
                                p_soft = intervalIndex * intervalSize;
                            }               
                            if (_resets) {                                        
                                size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
                                if (neighbor_index != SIZE_MAX) {
                                    // Edge exists, access its attributes
                                    auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;

                                    double error_probability = edge_it->error_probability;
                                    double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;

                                    pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
                                } else {
                                    throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
                                }
                            } else {
                                if (msmt < (distance-1)*(synd_rounds-1)) {
                                    double L = -std::log(p_soft/(1-p_soft));
                                    pm::add_edge(matching, msmt, msmt + 2 * (distance-1), empty_set, L, 0.5, "replace");
                                } else {
                                    size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
                                    if (neighbor_index != SIZE_MAX) {
                                        // Edge exists, access its attributes
                                        auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;

                                        double error_probability = edge_it->error_probability;
                                        double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;

                                        pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
                                    } else {
                                        throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
                                    }
                                }
                            }
                        }
                        if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
                            count_key += " ";
                        }  

                    }
                    auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false);
                    auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
                    auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
                    int actual_observable = (static_cast<int>(count_key.back()) - logical) % 2;

                    if (_detailed) {
                        ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                        result.error_details[shot] = errorDetail;
                    }
                    #pragma omp critical
                    {
                        if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
                            result.num_errors++; // Increment error count if they don't match
                            result.indices.push_back(shot);
                        }
                    }
                }
            }
            return result;
        }
    

    DetailedDecodeResult decode_IQ_1Dgauss(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping, 
        const std::map<int, std::map<std::string, float>> &gauss_params_dict, 
        bool _detailed) {

            DetailedDecodeResult result;
            result.num_errors = 0;

            for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
                Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
                auto counts = get_counts_1Dgauss(not_scaled_IQ_shot_matrix, qubit_mapping, gauss_params_dict, synd_rounds);
                std::string count_key = counts.begin()->first;

                soft_reweight_1Dgauss(matching, not_scaled_IQ_shot_matrix, synd_rounds,
                                      _resets, qubit_mapping, gauss_params_dict);

                auto det_syndromes = counts_to_det_syndr(count_key, _resets, false);
                auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());

                auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

                int actual_observable = (static_cast<int>(count_key[0]) - logical) % 2; // Convert first character to int and modulo 2
                // int actual_observable = (static_cast<int>(count_key[0]) - '0') % 2;  // Convert first character to int and modulo 2
                // Check if predicted_observables is not empty and compare the first element
                if (_detailed)
                {
                    ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                    result.error_details.push_back(errorDetail);
                }
                if (!predicted_observables.empty() && predicted_observables[0] != actual_observable)
                {
                    result.num_errors++; // Increment error count if they don't match
                    result.indices.push_back(shot);
                }
            }
            return result;

        }


    DetailedDecodeResult decode_IQ_kde(
        stim::DetectorErrorModel& detector_error_model,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        std::map<int, KDE_Result> kde_dict,
        bool _detailed,
        double relError,
        double absError,
        int nb_intervals) {

        DetailedDecodeResult result;
        result.num_errors = 0;
        result.error_details.resize(not_scaled_IQ_data.rows());

        std::set<size_t> empty_set;        
        int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

        UserGraph matching;

        #pragma omp parallel private(matching)
        {
            matching = detector_error_model_to_user_graph_private(detector_error_model);
            // DetailedDecodeResult localResult;
            // localResult.num_errors = 0;

            #pragma omp for nowait  
            for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
                Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
                std::string count_key;

                for (int msmt = 0; msmt < not_scaled_IQ_shot_matrix.cols(); ++msmt) { 
                    int qubit_idx = qubit_mapping.at(msmt);
                    auto &kde_entry = kde_dict.at(qubit_idx);
                    std::complex<double> not_scaled_point = not_scaled_IQ_data(shot, msmt);   

                    arma::mat query_point(2, 1); // 2 rows, 1 column
                    query_point(0, 0) = std::real(not_scaled_point); // real
                    query_point(1, 0) = std::imag(not_scaled_point); // imag 

                    query_point.row(0) -= kde_entry.scaler_mean[0]; // Element-wise subtraction
                    query_point.row(1) -= kde_entry.scaler_mean[1]; // Element-wise subtraction
                    query_point.row(0) /= kde_entry.scaler_stddev[0]; // Element-wise division
                    query_point.row(1) /= kde_entry.scaler_stddev[1]; // Element-wise division
                    
                    arma::vec estimations0(1);
                    arma::vec estimations1(1);

                    if (relError != -1) {
                        kde_entry.kde_0.RelativeError(relError);
                        kde_entry.kde_1.RelativeError(relError);
                    }
                    if (absError != -1) {
                        kde_entry.kde_0.AbsoluteError(absError);
                        kde_entry.kde_1.AbsoluteError(absError);
                    }

                    kde_entry.kde_0.Evaluate(query_point, estimations0);
                    kde_entry.kde_1.Evaluate(query_point, estimations1);

                    double p_small;
                    double p_big;

                    if (estimations0[0] > estimations1[0]) {
                        count_key += "0";    
                        p_small = estimations1[0];
                        p_big = estimations0[0];
                    } else {
                        count_key += "1";
                        p_small = estimations0[0];
                        p_big = estimations1[0];
                    }

                    if (msmt < (distance-1)*synd_rounds) {  
                        double p_soft = 1 / (1 + p_big/p_small);    

                        // round p_soft
                        if (nb_intervals != -1) {
                            double intervalSize = 0.5 / nb_intervals;
                            int intervalIndex = static_cast<int>(p_soft / intervalSize);
                            p_soft = intervalIndex * intervalSize;
                        }               
                        if (_resets) {                                        
                            size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
                            if (neighbor_index != SIZE_MAX) {
                                // Edge exists, access its attributes
                                auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;

                                double error_probability = edge_it->error_probability;
                                double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;

                                // std::cout << "" << std::endl;
                                // std::cout << "weight of edge (before):" << edge_it->weight << std::endl;
                                // std::cout << "err_proba of edge: " << error_probability << std::endl;
                                // std::cout << "" << std::endl;
                                // std::cout << "p_big, p_small" << p_big << ", " << p_small << std::endl;
                                // std::cout << "p_soft: " << p_soft << std::endl;
                                // std::cout << "new weight of edge:" << -std::log(p_tot/(1-p_tot)) << std::endl;

                                pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
                            } else {
                                throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
                            }
                        } else {
                            if (msmt < (distance-1)*(synd_rounds-1)) {
                                double L = -std::log(p_soft/(1-p_soft));
                                pm::add_edge(matching, msmt, msmt + 2 * (distance-1), empty_set, L, 0.5, "replace");
                            } else {
                                size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
                                if (neighbor_index != SIZE_MAX) {
                                    // Edge exists, access its attributes
                                    auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;

                                    double error_probability = edge_it->error_probability;
                                    double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;

                                    // std::cout << "weight of edge (before):" << edge_it->weight << std::endl;
                                    // std::cout << "err_proba of edge: " << error_probability << std::endl;
                                    // std::cout << "p_soft: " << p_soft << std::endl;
                                    // std::cout << "new weight of edge:" << -std::log(p_tot/(1-p_tot)) << std::endl;

                                    pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
                                } else {
                                    throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
                                }
                            }
                        }
                    }

                    if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
                        count_key += " ";
                    }            
                }

                auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false);
                auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
                auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
                int actual_observable = (static_cast<int>(count_key.back()) - logical) % 2;

                if (_detailed) {
                    ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                    // localResult.error_details.push_back(errorDetail);
                    result.error_details[shot] = errorDetail;
                }
                #pragma omp critical
                {
                    if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
                        result.num_errors++; // Increment error count if they don't match
                        result.indices.push_back(shot);
                    }
                }
            }

            // //Combine results
            // #pragma omp critical
            // {
            //     result.num_errors += localResult.num_errors;
            //     result.indices.insert(result.indices.end(), localResult.indices.begin(), localResult.indices.end());
            //     result.error_details.insert(result.error_details.end(), localResult.error_details.begin(), localResult.error_details.end());
            // }
        }
        return result;
    }

    // PBL: varying if repeated
    // DetailedDecodeResult decode_IQ_kde(
    //     stim::DetectorErrorModel& detector_error_model,
    //     const Eigen::MatrixXcd &not_scaled_IQ_data,
    //     int synd_rounds,
    //     int logical,
    //     bool _resets,
    //     const std::map<int, int> &qubit_mapping,
    //     std::map<int, KDE_Result> kde_dict,
    //     bool _detailed) {

    //     DetailedDecodeResult result;
    //     result.num_errors = 0; 

    //     std::set<size_t> empty_set;        
    //     int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

    //     #pragma omp parallel
    //     {
    //         DetailedDecodeResult localResult;
    //         localResult.num_errors = 0;

    //         #pragma omp for nowait        
    //         for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
    //             UserGraph matching = detector_error_model_to_user_graph_private(detector_error_model);
    //             Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
    //             std::string count_key;

    //             for (int msmt = 0; msmt < not_scaled_IQ_shot_matrix.cols(); ++msmt) { 
    //                 int qubit_idx = qubit_mapping.at(msmt);
    //                 auto &kde_entry = kde_dict.at(qubit_idx);
    //                 std::complex<double> not_scaled_point = not_scaled_IQ_data(shot, msmt);   

    //                 arma::mat query_point(2, 1); // 2 rows, 1 column
    //                 query_point(0, 0) = std::real(not_scaled_point); // real
    //                 query_point(1, 0) = std::imag(not_scaled_point); // imag 

    //                 query_point.row(0) -= kde_entry.scaler_mean[0]; // Element-wise subtraction
    //                 query_point.row(1) -= kde_entry.scaler_mean[1]; // Element-wise subtraction
    //                 query_point.row(0) /= kde_entry.scaler_stddev[0]; // Element-wise division
    //                 query_point.row(1) /= kde_entry.scaler_stddev[1]; // Element-wise division
                    
    //                 arma::vec estimations0(1);
    //                 arma::vec estimations1(1);
    //                 kde_entry.kde_0.Evaluate(query_point, estimations0);
    //                 kde_entry.kde_1.Evaluate(query_point, estimations1);

    //                 double p_small;
    //                 double p_big;

    //                 if (estimations0[0] > estimations1[0]) {
    //                     count_key += "0";    
    //                     p_big = estimations0[0];
    //                     p_small = estimations1[0];
    //                 } else {
    //                     count_key += "1";
    //                     p_small = estimations0[0];
    //                     p_big = estimations1[0];
    //                 }

    //                 if (msmt < (distance-1)*synd_rounds) {  
    //                     double p_soft = 1 / (1 + p_big/p_small);                   
    //                     if (_resets) {                                   
    //                             pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_soft/(1-p_soft)), p_soft, "independent");
    //                     } else {
    //                         if (msmt < (distance-1)*(synd_rounds-1)) {
    //                             pm::add_edge(matching, msmt, msmt + 2 * (distance-1), empty_set, -std::log(p_soft/(1-p_soft)), p_soft, "replace");
    //                         } else {                            
    //                             pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_soft/(1-p_soft)), p_soft, "independent");
    //                         }
    //                     }
    //                 }
    //                 if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
    //                     count_key += " ";
    //                 }            
    //             }
    //             // std::reverse(count_key.begin(), count_key.end()); // Reverse string (NOT NEEDED BECAUSE SLOWS DOWN)

    //             auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false);
    //             auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
    //             auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
    //             int actual_observable = (static_cast<int>(count_key[-1]) - logical) % 2;

    //             if (_detailed) {
    //                 ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
    //                 localResult.error_details.push_back(errorDetail);
    //             }
    //             if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
    //                 localResult.num_errors++; // Increment error count if they don't match
    //                 localResult.indices.push_back(shot);
    //             }
    //         }              
            
    //         // Combine results
    //         #pragma omp critical
    //         {
    //             result.num_errors += localResult.num_errors;
    //             result.indices.insert(result.indices.end(), localResult.indices.begin(), localResult.indices.end());
    //             result.error_details.insert(result.error_details.end(), localResult.error_details.begin(), localResult.error_details.end());
    //         }
    //     }
        
    //     return result;
    // }

    // PBL: not parrallelized
    // DetailedDecodeResult decode_IQ_kde(
    //     // UserGraph &matching,
    //     stim::DetectorErrorModel& detector_error_model,
    //     const Eigen::MatrixXcd &not_scaled_IQ_data,
    //     int synd_rounds,
    //     int logical,
    //     bool _resets,
    //     const std::map<int, int> &qubit_mapping,
    //     std::map<int, KDE_Result> kde_dict,
    //     bool _detailed) {

    //     DetailedDecodeResult result;
    //     result.num_errors = 0;
    //     UserGraph matching = detector_error_model_to_user_graph_private(detector_error_model);

    //     std::set<size_t> empty_set;        
    //     int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

    //     for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
    //         Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
    //         std::string count_key;

    //         for (int msmt = 0; msmt < not_scaled_IQ_shot_matrix.cols(); ++msmt) { 
    //             int qubit_idx = qubit_mapping.at(msmt);
    //             auto &kde_entry = kde_dict.at(qubit_idx);
    //             std::complex<double> not_scaled_point = not_scaled_IQ_data(shot, msmt);   

    //             arma::mat query_point(2, 1); // 2 rows, 1 column
    //             query_point(0, 0) = std::real(not_scaled_point); // real
    //             query_point(1, 0) = std::imag(not_scaled_point); // imag 

    //             query_point.row(0) -= kde_entry.scaler_mean[0]; // Element-wise subtraction
    //             query_point.row(1) -= kde_entry.scaler_mean[1]; // Element-wise subtraction
    //             query_point.row(0) /= kde_entry.scaler_stddev[0]; // Element-wise division
    //             query_point.row(1) /= kde_entry.scaler_stddev[1]; // Element-wise division
                
    //             arma::vec estimations0(1);
    //             arma::vec estimations1(1);
    //             kde_entry.kde_0.Evaluate(query_point, estimations0);
    //             kde_entry.kde_1.Evaluate(query_point, estimations1);

    //             double p_small;
    //             double p_big;

    //             if (estimations0[0] > estimations1[0]) {
    //                 count_key += "0";    
    //                 p_small = estimations1[0];
    //                 p_big = estimations0[0];
    //             } else {
    //                 count_key += "1";
    //                 p_small = estimations0[0];
    //                 p_big = estimations1[0];
    //             }

    //             if (msmt < (distance-1)*synd_rounds) {  
    //                 double p_soft = 1 / (1 + p_big/p_small);                   
    //                 if (_resets) {                                        
    //                     size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
    //                     if (neighbor_index != SIZE_MAX) {
    //                         // Edge exists, access its attributes
    //                         auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;

    //                         double error_probability = edge_it->error_probability;
    //                         double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;

    //                         pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
    //                     } else {
    //                         throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
    //                     }
    //                 } else {
    //                     if (msmt < (distance-1)*(synd_rounds-1)) {
    //                         double L = -std::log(p_soft/(1-p_soft));
    //                         pm::add_edge(matching, msmt, msmt + 2 * (distance-1), empty_set, L, 0.5, "replace");
    //                     } else {
    //                         size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
    //                         if (neighbor_index != SIZE_MAX) {
    //                             // Edge exists, access its attributes
    //                             auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;

    //                             double error_probability = edge_it->error_probability;
    //                             double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;

    //                             pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
    //                         } else {
    //                             throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
    //                         }
    //                     }
    //                 }
    //             }

    //             if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
    //                 count_key += " ";
    //             }            
    //         }

    //         // std::cout << "count_key 1:_" << count_key << "_" << std::endl;

    //         // std::reverse(count_key.begin(), count_key.end()); // Reverse string (NOT NEEDED BECAUSE SLOWS DOWN)

    //         // std::cout << "count_key 2:_" << count_key << "_" << std::endl;

    //         // std::cout << "count_key[0]:_" << count_key[0] << "_" << std::endl;
    //         // std::cout << "count_key[-1]:_" << count_key.back() << "_" << std::endl;=

    //         auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false);
    //         auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
    //         auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
    //         int actual_observable = (static_cast<int>(count_key.back()) - logical) % 2;

    //         if (_detailed) {
    //             ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
    //             result.error_details.push_back(errorDetail);
    //         }
    //         if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
    //             result.num_errors++; // Increment error count if they don't match
    //             result.indices.push_back(shot);
    //         }
    //     }

    //     return result;
    // }

    // PBL: profiling std::couts
    // DetailedDecodeResult decode_IQ_kde(
    //     UserGraph &matching,
    //     const Eigen::MatrixXcd &not_scaled_IQ_data,
    //     int synd_rounds,
    //     int logical,
    //     bool _resets,
    //     const std::map<int, int> &qubit_mapping,
    //     std::map<int, KDE_Result> kde_dict,
    //     bool _detailed) {

    //     DetailedDecodeResult result;
    //     result.num_errors = 0;

    //     std::set<size_t> empty_set;

    //     // Start of profiling
    //     auto start_total = std::chrono::high_resolution_clock::now();
        
    //     int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

    //     for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
    //         // Start timing for this shot
    //         auto start_shot = std::chrono::high_resolution_clock::now();

    //         Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);

    //         // Measure time for get_counts_kde
    //         auto start_get_counts = std::chrono::high_resolution_clock::now();

    //         std::string count_key;
    //         for (int msmt = 0; msmt < not_scaled_IQ_shot_matrix.cols(); ++msmt) { 
    //             int qubit_idx = qubit_mapping.at(msmt);
    //             auto &kde_entry = kde_dict.at(qubit_idx);
    //             std::complex<double> not_scaled_point = not_scaled_IQ_data(shot, msmt);   

    //             arma::mat query_point(2, 1); // 2 rows, 1 column
    //             query_point(0, 0) = std::real(not_scaled_point); // real
    //             query_point(1, 0) = std::imag(not_scaled_point); // imag 

    //             query_point.row(0) -= kde_entry.scaler_mean[0]; // Element-wise subtraction
    //             query_point.row(1) -= kde_entry.scaler_mean[1]; // Element-wise subtraction
    //             query_point.row(0) /= kde_entry.scaler_stddev[0]; // Element-wise division
    //             query_point.row(1) /= kde_entry.scaler_stddev[1]; // Element-wise division
                
    //             arma::vec estimations0(1);
    //             arma::vec estimations1(1);
    //             kde_entry.kde_0.Evaluate(query_point, estimations0);
    //             kde_entry.kde_1.Evaluate(query_point, estimations1);

    //             double p_small;
    //             double p_big;

    //             if (estimations0[0] > estimations1[0]) {
    //                 count_key += "0";    
    //                 p_small = estimations1[0];
    //                 p_big = estimations0[0];
    //             } else {
    //                 count_key += "1";
    //                 p_small = estimations0[0];
    //                 p_big = estimations1[0];
    //             }

    //             if (msmt < (distance-1)*synd_rounds) {  
    //                 double p_soft = 1 / (1 + p_big/p_small);                   
    //                 if (_resets) {                                        
    //                     size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
    //                     if (neighbor_index != SIZE_MAX) {
    //                         // Edge exists, access its attributes
    //                         auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;

    //                         double error_probability = edge_it->error_probability;
    //                         double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;

    //                         pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
    //                     } else {
    //                         throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
    //                     }
    //                 } else {
    //                     if (msmt < (distance-1)*(synd_rounds-1)) {
    //                         double L = -std::log(p_soft/(1-p_soft));
    //                         pm::add_edge(matching, msmt, msmt + 2 * (distance-1), empty_set, L, 0.5, "replace");
    //                     } else {
    //                         size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
    //                         if (neighbor_index != SIZE_MAX) {
    //                             // Edge exists, access its attributes
    //                             auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;

    //                             double error_probability = edge_it->error_probability;
    //                             double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;

    //                             pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
    //                         } else {
    //                             throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
    //                         }
    //                     }
    //                 }
    //             }

    //             if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
    //                 count_key += " ";
    //             }            
    //         }
    //         // std::reverse(count_key.begin(), count_key.end()); // Reverse string

    //         auto end_get_counts = std::chrono::high_resolution_clock::now();
    //         auto get_counts_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_get_counts - start_get_counts);

    //         // Measure time for counts_to_det_syndr
    //         auto start_counts_to_det_syndr = std::chrono::high_resolution_clock::now();
    //         auto det_syndromes = counts_to_det_syndr(count_key, _resets, false);
    //         auto end_counts_to_det_syndr = std::chrono::high_resolution_clock::now();
    //         auto counts_to_det_syndr_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_counts_to_det_syndr - start_counts_to_det_syndr);

    //         auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());

    //         // Measure time for decode
    //         auto start_decode = std::chrono::high_resolution_clock::now();
    //         auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
    //         auto end_decode = std::chrono::high_resolution_clock::now();
    //         auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_decode - start_decode);

    //         int actual_observable = (static_cast<int>(count_key[0]) - logical) % 2;

    //         // Stop timing for this shot
    //         auto end_shot = std::chrono::high_resolution_clock::now();

    //         // Calculate the time for this shot
    //         auto shot_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_shot - start_shot);

    //         // Print the time for each part
    //         std::cout << "Shot " << shot << " timing:" << std::endl;
    //         std::cout << "get_counts_kde: " << get_counts_duration.count() << " milliseconds" << std::endl;
    //         std::cout << "counts_to_det_syndr: " << counts_to_det_syndr_duration.count() << " milliseconds" << std::endl;
    //         std::cout << "decode: " << decode_duration.count() << " milliseconds" << std::endl;
    //         std::cout << "Total shot execution time: " << shot_duration.count() << " milliseconds" << std::endl;

    //         if (_detailed) {
    //             ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
    //             result.error_details.push_back(errorDetail);
    //         }
    //         if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
    //             result.num_errors++; // Increment error count if they don't match
    //             result.indices.push_back(shot);
    //         }
    //     }

    //     // Stop timing for the entire function
    //     auto end_total = std::chrono::high_resolution_clock::now();

    //     // Calculate the total time for the function
    //     auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);

    //     // Print the total time for the function
    //     std::cout << "Total execution time: " << total_duration.count() << " milliseconds" << std::endl;

    //     return result;
    // }


    
    // // has cummulative reweighting problem + extremely slow (slower than python)
    // DetailedDecodeResult decode_IQ_kde(
    //     UserGraph &matching,
    //     const Eigen::MatrixXcd &not_scaled_IQ_data,
    //     int synd_rounds,
    //     int logical,
    //     bool _resets,
    //     const std::map<int, int> &qubit_mapping, 
    //     std::map<int, KDE_Result> kde_dict,
    //     bool _detailed) {
            
    //         DetailedDecodeResult result;
    //         result.num_errors = 0;

    //         // Start of profiling
    //         auto start_total = std::chrono::high_resolution_clock::now();

    //         for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
    //             Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);

    //             auto counts = get_counts_kde(not_scaled_IQ_shot_matrix, qubit_mapping, kde_dict, synd_rounds);
    //             std::string count_key = counts.begin()->first;

    //             soft_reweight_kde(matching, not_scaled_IQ_shot_matrix, synd_rounds,
    //                                   _resets, qubit_mapping, kde_dict);

    //             auto det_syndromes = counts_to_det_syndr(count_key, _resets, false);
    //             auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());

    //             auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

    //             int actual_observable = (static_cast<int>(count_key[0]) - logical) % 2; // Convert first character to int and modulo 2
    //             // int actual_observable = (static_cast<int>(count_key[0]) - '0') % 2;  // Convert first character to int and modulo 2
    //             // Check if predicted_observables is not empty and compare the first element
    //             if (_detailed)
    //             {
    //                 ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
    //                 result.error_details.push_back(errorDetail);
    //             }
    //             if (!predicted_observables.empty() && predicted_observables[0] != actual_observable)
    //             {
    //                 result.num_errors++; // Increment error count if they don't match
    //                 result.indices.push_back(shot);
    //             }
    //         }
    //         return result;
    //     }



    DetailedDecodeResult decode_IQ_shots_flat(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, GridData> &kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
        bool _detailed)
    {

        DetailedDecodeResult result;
        result.num_errors = 0;

        reweight_edges_to_one(matching);
        for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot)
        {
            Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
            auto counts = get_counts(not_scaled_IQ_shot_matrix, qubit_mapping, kde_grid_dict, scaler_params_dict, synd_rounds);
            std::string count_key = counts.begin()->first;

            auto det_syndromes = counts_to_det_syndr(count_key, _resets, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());

            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

            int actual_observable = (static_cast<int>(count_key[0]) - logical) % 2; // Convert first character to int and modulo 2
            // Check if predicted_observables is not empty and compare the first element
            if (_detailed)
            {
                ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                result.error_details.push_back(errorDetail);
            }
            if (!predicted_observables.empty() && predicted_observables[0] != actual_observable)
            {
                result.num_errors++; // Increment error count if they don't match
                result.indices.push_back(shot);
            }
        }

        return result;
    }

    DetailedDecodeResult decode_IQ_shots_flat_informed(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, GridData> &kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
        float p_data, float p_mixed, float p_meas, float common_measure,
        bool _detailed, bool _ntnn_edges)
    {

        DetailedDecodeResult result;
        result.num_errors = 0;

        // Distance
        int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes
        reweight_edges_informed(matching, distance, p_data, p_mixed, p_meas, common_measure, _ntnn_edges);
        for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot)
        {
            Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
            auto counts = get_counts(not_scaled_IQ_shot_matrix, qubit_mapping, kde_grid_dict, scaler_params_dict, synd_rounds);
            std::string count_key = counts.begin()->first;

            auto det_syndromes = counts_to_det_syndr(count_key, _resets, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());

            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

            int actual_observable = (static_cast<int>(count_key[0]) - logical) % 2; // Convert first character to int and modulo 2
            // Check if predicted_observables is not empty and compare the first element
            if (_detailed)
            {
                ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                result.error_details.push_back(errorDetail);
            }
            if (!predicted_observables.empty() && predicted_observables[0] != actual_observable)
            {
                result.num_errors++; // Increment error count if they don't match
                result.indices.push_back(shot);
            }
        }

        return result;
    }

    DetailedDecodeResult decode_IQ_shots_flat_err_probs(
        UserGraph &matching,
        int logical,
        const std::map<std::string, size_t> &counts_tot,
        bool _resets,
        const std::string &method,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, GridData> &kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
        bool _detailed)
    {

        DetailedDecodeResult result;
        result.num_errors = 0;

        // Distance
        int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes
        reweight_edges_based_on_error_probs(matching, counts_tot, _resets, method);

        // Loop over shots and decode each one
        for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot)
        {
            // Process the IQ data for each shot
            Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
            auto counts = get_counts(not_scaled_IQ_shot_matrix, qubit_mapping, kde_grid_dict, scaler_params_dict, synd_rounds);

            // Continue with the decoding process as in the original function
            std::string count_key = counts.begin()->first;
            auto det_syndromes = counts_to_det_syndr(count_key, _resets, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

            int actual_observable = (static_cast<int>(count_key[0]) - logical) % 2; // Convert first character to int and modulo 2
            if (_detailed)
            {
                ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                result.error_details.push_back(errorDetail);
            }
            if (!predicted_observables.empty() && predicted_observables[0] != actual_observable)
            {
                result.num_errors++; // Increment error count if they don't match
                result.indices.push_back(shot);
            }
        }

        return result;
    }

    DetailedDecodeResult decode_IQ_shots_no_reweighting(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, GridData> &kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
        bool _detailed)
    {

        DetailedDecodeResult result;
        result.num_errors = 0;

        // Loop over shots and decode each one
        for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot)
        {
            // Process the IQ data for each shot
            Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);
            auto counts = get_counts(not_scaled_IQ_shot_matrix, qubit_mapping, kde_grid_dict, scaler_params_dict, synd_rounds);

            // Continue with the decoding process as in the original function
            std::string count_key = counts.begin()->first;
            auto det_syndromes = counts_to_det_syndr(count_key, _resets, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);

            int actual_observable = (static_cast<int>(count_key[0]) - logical) % 2; // Convert first character to int and modulo 2
            if (!predicted_observables.empty() && predicted_observables[0] != actual_observable)
            {
                result.num_errors++; // Increment error count if they don't match
                result.indices.push_back(shot);
                if (_detailed)
                {
                    ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                    result.error_details.push_back(errorDetail);
                }
            }
        }

        return result;
    }

}

void processGraph_test(pm::UserGraph &graph)
{
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
