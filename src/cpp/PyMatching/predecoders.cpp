#include "predecoders.h"


#include <omp.h>



namespace pd {

    PreDecodeResult decode_time_nn_predecode_grid(
        stim::DetectorErrorModel detector_error_model,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, GridData> &kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
        bool _detailed,
        float threshold,
        bool _ntnn_edges) {

            pm::DetailedDecodeResult result;
            result.num_errors = 0;
            result.error_details.resize(not_scaled_IQ_data.rows());

            std::vector<int> nb_rm_edges_vec(not_scaled_IQ_data.rows(), 0);

            std::set<size_t> empty_set;        
            int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes
            
            pm::UserGraph matching;
            #pragma omp parallel private(matching)
            {
                matching = pm::detector_error_model_to_user_graph_private(detector_error_model);
                if (_ntnn_edges) {
                    pm::reweight_edges_informed(matching, distance, -1, -1, -1, -1, _ntnn_edges); // -1 bcs we don't use the weights
                }
                #pragma omp for nowait  
                for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) {
                    Eigen::MatrixXcd not_scaled_IQ_shot_matrix = not_scaled_IQ_data.row(shot);

                    std::vector<int> counts;
                    std::vector<int> corrected_counts;
                    std::vector<int> det_syndromes;

                    int corrected_bit;
                    int detector_bit;

                    int nb_rm_edges = 0;

                    for (int msmt = 0; msmt < not_scaled_IQ_shot_matrix.cols(); ++msmt) { 
                        int qubit_idx = qubit_mapping.at(msmt);
                        std::complex<double> not_scaled_point = not_scaled_IQ_data(shot, msmt);
                        auto &grid_data = kde_grid_dict.at(qubit_idx);

                        const auto& [real_params, imag_params] = scaler_params_dict.at(qubit_idx);
                        double real_scaled = (std::real(not_scaled_point) - real_params.first) / real_params.second;
                        double imag_scaled = (std::imag(not_scaled_point) - imag_params.first) / imag_params.second;
                        Eigen::Vector2d scaled_point = {real_scaled, imag_scaled};

                        auto [outcome, density0, density1] = grid_lookup(scaled_point, grid_data);

                        // 1) Get the detector bit
                        if (msmt < distance-1) { // first row of measurement 
                            detector_bit = outcome;
                            if (!_resets) {
                                corrected_bit = outcome;
                            }
                            det_syndromes.push_back(detector_bit);
                            corrected_counts.push_back(corrected_bit);
                        }
                        else if (msmt > (distance-1)*synd_rounds) { // last row of data measurements + 1 (== >)
                            // std::cout << "msmt: " << msmt << std::endl;
                            // std::cout << "outcome: " << outcome << std::endl;
                            // std::cout << "counts.back(): " << counts.back() << std::endl;
                            corrected_bit = (outcome + counts.back()) % 2; // using the corrected_bit structure (incomprehensible but efficient)
                            if (!_resets) {
                                detector_bit = (corrected_bit + corrected_counts[msmt-1-(distance-1)])  % 2; // -1 because we skip 1 measurement
                            }
                            else {
                                detector_bit = (corrected_bit + counts[msmt-1-(distance-1)]) % 2; // -1 because we skip 1 measurement
                            }
                            det_syndromes.push_back(detector_bit);
                            corrected_counts.push_back(corrected_bit);
                        }
                        else if (msmt < (distance-1)*synd_rounds) { // Middle rows of measurements (without the first data msmt)
                            if (!_resets) {
                                corrected_bit = (outcome + counts[msmt-(distance-1)]) % 2;
                                detector_bit = (corrected_bit + corrected_counts[msmt-(distance-1)])  % 2;
                            }
                            else {
                                detector_bit = (outcome + counts[msmt-(distance-1)]) % 2;
                            }
                            det_syndromes.push_back(detector_bit);
                            corrected_counts.push_back(corrected_bit);
                        }
                        
                        // Push back the count
                        counts.push_back(outcome);

                        // pointer to the needed elements of the syndrome
                        int* det_bit_ptr = &det_syndromes.back();
                        
                        // 3) Mark suspicious detector bits (if not already removed because of previous detector bit = 2)
                        if (*det_bit_ptr == 1 and msmt < (distance-1)*synd_rounds) { // up to the last row of syndrom measurements

                        // TODO: for the first data msmt this is checked twice if det_bit_ptr is 1 but below threshold! (the overhead should be minimal)
                            
                            // GET P_SOFT
                            double p_small;
                            double p_big;
                            if (outcome == 0) {
                                p_small = std::exp(density1);
                                p_big = std::exp(density0);
                            } else {
                                p_small = std::exp(density0);
                                p_big = std::exp(density1);
                            }
                            
                            double p_soft = 1 / (1 + p_big/p_small);
                            // std::cout << "p_soft: " << p_soft << std::endl;


                            // Mark suspicious detector bits
                            if (p_soft > threshold) {
                                *det_bit_ptr = 2;
                            }
                        }
                    }

                    // // print the counts
                    // std::cout << std::endl;
                    // std::cout << "Shot " << shot << " has counts: " << std::endl;
                    
                    // for (int i = 0; i < counts.size(); ++i) {
                    //     std::cout << counts[i] << " ";
                    // }
                    // std::cout << std::endl;

                    // // print the corrected counts
                    // // std::cout << std::endl;
                    // std::cout << "Shot " << shot << " has corrected counts: " << std::endl;
                    // for (int i = 0; i < corrected_counts.size(); ++i) {
                    //     std::cout << corrected_counts[i] << " ";
                    // }
                    // std::cout << std::endl;

                    // // print the syndromes
                    // // std::cout << std::endl;
                    // std::cout << "Shot (bf sus corrections) " << shot << " has syndromes: " << std::endl;
                    // for (int i = 0; i < det_syndromes.size(); ++i) {
                    //     std::cout << det_syndromes[i] << " ";
                    // }
                    // std::cout << std::endl;

                    // 3) Correct detector syndromes with suspicious bits (Iterate through the syndromes)
                    for (int det_idx = 0; det_idx < det_syndromes.size(); ++det_idx) {
                        if (det_idx >= distance-1) { // Starting the second row of nodes

                            int* det_bit_ptr = &det_syndromes[det_idx];
                            int* prev_bit_ptr = &det_syndromes[det_idx-(distance-1)];

                            if (!_resets) { // No Resets
                                if (det_idx >= 2*(distance-1)) { // AFTER the second row (bcs NTNN edges)
                                    int* prev_prev_bit_ptr = &det_syndromes[det_idx-2*(distance-1)];

                                    if ((*det_bit_ptr == 1 || *det_bit_ptr == 2) && *prev_prev_bit_ptr == 2) { // if the previous NTNN detector bit was suspicious
                                        *det_bit_ptr = 0;
                                        *prev_prev_bit_ptr= 0;
                                        nb_rm_edges++;
                                    }
                                    else if (*prev_prev_bit_ptr == 2) { // rm the suspicious label from the remaining NTNN bit
                                        *prev_prev_bit_ptr = 1;
                                    }

                                    if (det_idx >=(distance-1)*synd_rounds) { // For data detector bits (bcs they have distance 1 soft edges)
                                        if (*det_bit_ptr == 1 and *prev_bit_ptr==2) { // if the previous detector bit was suspicious
                                            *det_bit_ptr = 0;
                                            *prev_bit_ptr= 0;
                                            nb_rm_edges++;
                                        }
                                        else if (*prev_bit_ptr == 2) { // rm the suspicious label from the remaining bits
                                            *prev_bit_ptr = 1;
                                        }
                                    }
                                }
                            }
                            else { // With Resets
                                if ((*det_bit_ptr == 1 || *det_bit_ptr == 2) && *prev_bit_ptr == 2) { // if the previous detector bit was suspicious
                                    *det_bit_ptr = 0;
                                    *prev_bit_ptr= 0;
                                    nb_rm_edges++;
                                }
                                else if (*prev_bit_ptr == 2) { // rm the suspicious bit
                                    *prev_bit_ptr = 1;
                                }
                            }
                        }
                    }

                    // // print the syndromes
                    // // std::cout << std::endl;
                    // std::cout << "Shot (af sus corrections) " << shot << " has syndromes: " << std::endl;
                    // for (int i = 0; i < det_syndromes.size(); ++i) {
                    //     std::cout << det_syndromes[i] << " ";
                    // }
                    // std::cout << std::endl;
                    // std::cout << "Shot " << shot << " has " << nb_rm_edges << " removed edges" << std::endl;
                    // std::cout << std::endl;



                    // 4) Decode the shot
                    auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
                    auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
                    int actual_observable = (static_cast<int>(counts.back()) - logical) % 2;

                    #pragma omp critical
                    nb_rm_edges_vec[shot] = nb_rm_edges;

                    if (_detailed) {
                        pm::ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                        result.error_details[shot] = errorDetail;
                    }
                    {
                        if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
                            result.num_errors++; // Increment error count if they don't match
                            result.indices.push_back(shot);
                        }
                    }

                }
            }
            PreDecodeResult preDecodeResult;
            preDecodeResult.decode_result = result;
            preDecodeResult.nb_rm_edges = nb_rm_edges_vec;
            return preDecodeResult;
        }










}