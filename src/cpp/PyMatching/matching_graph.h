#ifndef MATCHING_H
#define MATCHING_H


#include <Eigen/Dense>  
#include "../Probabilities/probabilities.h"
#include "pymatching/sparse_blossom/driver/user_graph.h" // Include necessary headers for declarations

namespace pm {
    void soft_reweight_pymatching(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict, // Adjusted to hold pairs of pairs
        float p_data = -1, float p_mixed = -1, float common_measure = -1);

    void reweight_edges_to_one(UserGraph &matching);

    void reweight_edges_informed(
        UserGraph &matching,  float distance, 
        float p_data, float p_mixed, float p_meas, 
        float common_measure);

    int decode_IQ_shots(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict, 
        float p_data, float p_mixed, float common_measure = -1);
    
    int decode_IQ_shots_flat(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict);

    int decode_IQ_shots_flat_informed(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict,
        float p_data, float p_mixed, float p_meas, float common_measure = -1);
}


void processGraph_test(pm::UserGraph& graph);

#endif // MATCHING_H