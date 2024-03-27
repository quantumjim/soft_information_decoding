#include "convdecoders.h"
    

pm::DetailedDecodeResult decodeConvertorSoft(
    stim::DetectorErrorModel& detector_error_model,
    Eigen::MatrixXi comparisonMatrix,
    Eigen::MatrixXd pSoftMatrix,
    int synd_rounds,
    int logical,
    bool _resets,
    bool _detailed) {
        
    pm::DetailedDecodeResult result;
    result.num_errors = 0;

    std::set<size_t> empty_set;        
    int distance = (comparisonMatrix.cols() + synd_rounds) / (synd_rounds + 1); // Adapted for RepCodes

    pm::UserGraph matching;

    #pragma omp parallel private(matching)
    {
        matching = pm::detector_error_model_to_user_graph_private(detector_error_model);

        #pragma omp for nowait  
        for (int shot = 0; shot < comparisonMatrix.rows(); ++shot) {
            std::string count_key;
            for (int msmt = 0; msmt < comparisonMatrix.cols(); ++msmt) {

                // Counts
                int measurementResult = comparisonMatrix(shot, msmt);
                if (measurementResult == 1) {
                    count_key += "1";
                } else {
                    count_key += "0";
                }
                if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
                    count_key += " ";
                }  

                // Reweighting
                if (msmt < (distance-1)*synd_rounds) {
                    double p_soft = pSoftMatrix(shot, msmt);
                    if (_resets) {
                        size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
                        if (neighbor_index != SIZE_MAX) {
                            auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;
                            double error_probability = edge_it->error_probability;
                            double p_tot = p_soft * (1-error_probability) + (1-p_soft) * error_probability;
                            pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
                        } else {
                            throw std::runtime_error("Edge does not exist between: " + std::to_string(msmt) + " and " + std::to_string(msmt + (distance-1)));
                        }
                    } else {
                        if (msmt < (distance-1)*(synd_rounds-1)) {
                            double L = -std::log(p_soft/(1-p_soft));
                            pm::add_edge(matching, msmt, msmt + 2 * (distance-1), empty_set, L, 0.5, "replace");
                        } else {
                            size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
                            if (neighbor_index != SIZE_MAX) {
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
            }

            // Decoding
            auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
            int actual_observable = (static_cast<int>(count_key.back()) - logical) % 2;

            // Result saving
            if (_detailed) {
                pm::ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
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

std::tuple<pm::DetailedDecodeResult, pm::DetailedDecodeResult> decodeConvertorAll(
    stim::DetectorErrorModel& detector_error_model,
    Eigen::MatrixXi comparisonMatrix,
    Eigen::MatrixXd pSoftMatrix,
    int synd_rounds,
    int logical,
    bool _resets) {
        
    pm::DetailedDecodeResult result_soft;
    result_soft.num_errors = 0;

    pm::DetailedDecodeResult result_hard;
    result_hard.num_errors = 0;

    std::set<size_t> empty_set;        
    int distance = (comparisonMatrix.cols() + synd_rounds) / (synd_rounds + 1); // Adapted for RepCodes

    pm::UserGraph matching;

    #pragma omp parallel private(matching)
    {
        matching = pm::detector_error_model_to_user_graph_private(detector_error_model);

        #pragma omp for nowait  
        for (int shot = 0; shot < comparisonMatrix.rows(); ++shot) {
            std::string count_key;
            for (int msmt = 0; msmt < comparisonMatrix.cols(); ++msmt) {

                // Counts
                int measurementResult = comparisonMatrix(shot, msmt);
                if (measurementResult == 1) {
                    count_key += "1";
                } else {
                    count_key += "0";
                }
                if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
                    count_key += " ";
                }  

                // Reweighting
                if (msmt < (distance-1)*synd_rounds) {
                    double p_soft = pSoftMatrix(shot, msmt);
                    if (_resets) {
                        size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
                        if (neighbor_index != SIZE_MAX) {
                            auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;
                            double error_probability = edge_it->error_probability;
                            double p_tot = p_soft * (1-error_probability) + (1-p_soft) * error_probability;
                            pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
                        } else {
                            throw std::runtime_error("Edge does not exist between: " + std::to_string(msmt) + " and " + std::to_string(msmt + (distance-1)));
                        }
                    } else {
                        if (msmt < (distance-1)*(synd_rounds-1)) {
                            double L = -std::log(p_soft/(1-p_soft));
                            pm::add_edge(matching, msmt, msmt + 2 * (distance-1), empty_set, L, 0.5, "replace");
                        } else {
                            size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
                            if (neighbor_index != SIZE_MAX) {
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
            }

            // Decoding
            auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false);
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
            auto [predicted_observables_soft, rescaled_weight] = decode(matching, detectionEvents);

            // Reset the graph and redecode
            matching = pm::detector_error_model_to_user_graph_private(detector_error_model); 
            auto [predicted_observables_hard, rescaled_weight_hard] = decode(matching, detectionEvents);

            int actual_observable = (static_cast<int>(count_key.back()) - logical) % 2;

            // Result saving
            #pragma omp critical
            {
                if (!predicted_observables_soft.empty() && predicted_observables_soft[0] != actual_observable) {
                    result_soft.num_errors++; // Increment error count if they don't match
                    result_soft.indices.push_back(shot);
                }
                if (!predicted_observables_hard.empty() && predicted_observables_hard[0] != actual_observable) {
                    result_hard.num_errors++; // Increment error count if they don't match
                    result_hard.indices.push_back(shot);
                }
            }
        }
    }
    return std::make_tuple(result_soft, result_hard);
}



///////////// TIMED VERSION //////////////////

// pm::DetailedDecodeResult decodeConvertorSoft(
//     stim::DetectorErrorModel& detector_error_model,
//     Eigen::MatrixXi comparisonMatrix,
//     Eigen::MatrixXd pSoftMatrix,
//     int synd_rounds,
//     int logical,
//     bool _resets,
//     bool _detailed) {
        
//     pm::DetailedDecodeResult result;
//     result.num_errors = 0;

//     std::set<size_t> empty_set;        
//     int distance = (comparisonMatrix.cols() + synd_rounds) / (synd_rounds + 1); // Adapted for RepCodes

//     pm::UserGraph matching;

//     // Timing variables
//     auto start_total = std::chrono::steady_clock::now();

//     #pragma omp parallel private(matching)
//     {
//         matching = pm::detector_error_model_to_user_graph_private(detector_error_model);

//         #pragma omp for nowait  
//         for (int shot = 0; shot < comparisonMatrix.rows(); ++shot) {
//             auto start_shot = std::chrono::steady_clock::now();
//             std::string count_key;
//             std::chrono::duration<double> counts_time_shot(0);
//             std::chrono::duration<double> reweighting_time_shot(0);
//             for (int msmt = 0; msmt < comparisonMatrix.cols(); ++msmt) {

//                 // Timing for counts
//                 auto start_counts = std::chrono::steady_clock::now();

//                 // Counts
//                 int measurementResult = comparisonMatrix(shot, msmt);
//                 if (measurementResult == 1) {
//                     count_key += "1";
//                 } else {
//                     count_key += "0";
//                 }
//                 if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
//                     count_key += " ";
//                 }  

//                 // Timing for counts
//                 auto end_counts = std::chrono::steady_clock::now();
//                 counts_time_shot += end_counts - start_counts;

//                 // Timing for reweighting
//                 auto start_reweighting = std::chrono::steady_clock::now();

//                 // Reweighting
//                 if (msmt < (distance-1)*synd_rounds) {
//                     double p_soft = pSoftMatrix(shot, msmt);
//                     if (_resets) {
//                         size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
//                         if (neighbor_index != SIZE_MAX) {
//                             auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;
//                             double error_probability = edge_it->error_probability;
//                             double p_tot = p_soft * (1-error_probability) + (1-p_soft) * error_probability;
//                             pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
//                         } else {
//                             throw std::runtime_error("Edge does not exist between: " + std::to_string(msmt) + " and " + std::to_string(msmt + (distance-1)));
//                         }
//                     } else {
//                         if (msmt < (distance-1)*(synd_rounds-1)) {
//                             double L = -std::log(p_soft/(1-p_soft));
//                             pm::add_edge(matching, msmt, msmt + 2 * (distance-1), empty_set, L, 0.5, "replace");
//                         } else {
//                             size_t neighbor_index = matching.nodes[msmt].index_of_neighbor(msmt + (distance-1));
//                             if (neighbor_index != SIZE_MAX) {
//                                 auto edge_it = matching.nodes[msmt].neighbors[neighbor_index].edge_it;
//                                 double error_probability = edge_it->error_probability;
//                                 double p_tot = p_soft*(1-error_probability) + (1-p_soft)*error_probability;
//                                 pm::add_edge(matching, msmt, msmt + (distance-1), empty_set, -std::log(p_tot/(1-p_tot)), error_probability, "replace");
//                             } else {
//                                 throw std::runtime_error("Edge does not exist between:" + std::to_string(msmt) +  " and " + std::to_string(msmt + (distance-1)));
//                             }
//                         }
//                     }
//                 }

//                 // Timing for reweighting
//                 auto end_reweighting = std::chrono::steady_clock::now();
//                 reweighting_time_shot += end_reweighting - start_reweighting;
//             }

//             // Timing for decoding
//             auto start_decoding = std::chrono::steady_clock::now();

//             // Decoding
//             auto start_det_syndromes = std::chrono::steady_clock::now();
//             auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false);
//             auto end_det_syndromes = std::chrono::steady_clock::now();
//             std::chrono::duration<double> det_syndromes_time = end_det_syndromes - start_det_syndromes;

//             auto start_detectionEvents = std::chrono::steady_clock::now();
//             auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
//             auto end_detectionEvents = std::chrono::steady_clock::now();
//             std::chrono::duration<double> detectionEvents_time = end_detectionEvents - start_detectionEvents;

//             auto start_decode = std::chrono::steady_clock::now();
//             auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
//             auto end_decode = std::chrono::steady_clock::now();
//             std::chrono::duration<double> decode_time = end_decode - start_decode;

//             int actual_observable = (static_cast<int>(count_key.back()) - logical) % 2;

//             // Result saving
//             if (_detailed) {
//                 pm::ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
//                 result.error_details[shot] = errorDetail;
//             }
//             #pragma omp critical
//             {
//                 if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
//                     result.num_errors++; // Increment error count if they don't match
//                     result.indices.push_back(shot);
//                 }
//             }

//             // Timing for decoding
//             auto end_decoding = std::chrono::steady_clock::now();
//             std::chrono::duration<double> decoding_time = end_decoding - start_decoding;
//             std::cout << "Counts time for shot " << shot << ": " << std::scientific << counts_time_shot.count() << " seconds" << std::endl;
//             std::cout << "Reweighting time for shot " << shot << ": " << std::scientific << reweighting_time_shot.count() << " seconds" << std::endl;
//             std::cout << "Det syndromes time for shot " << shot << ": " << std::scientific << det_syndromes_time.count() << " seconds" << std::endl;
//             std::cout << "Detection events time for shot " << shot << ": " << std::scientific << detectionEvents_time.count() << " seconds" << std::endl;
//             std::cout << "Decode time for shot " << shot << ": " << std::scientific << decode_time.count() << " seconds" << std::endl;
//             std::cout << "Decoding time for shot " << shot << ": " << std::scientific << decoding_time.count() << " seconds" << std::endl;
//             auto end_shot = std::chrono::steady_clock::now();
//             std::chrono::duration<double> shot_time = end_shot - start_shot;
//             std::cout << "Total time for shot " << shot << ": " << std::scientific << shot_time.count() << " seconds" << std::endl;
//         }
//     }

//     // Timing for total
//     auto end_total = std::chrono::steady_clock::now();
//     std::chrono::duration<double> total_time = end_total - start_total;
//     std::cout << "Total time: " << std::scientific << total_time.count() << " seconds" << std::endl;

//     return result;
// }