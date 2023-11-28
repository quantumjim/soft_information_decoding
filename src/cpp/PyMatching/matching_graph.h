#ifndef MATCHING_H
#define MATCHING_H


#include <Eigen/Dense>  
#include "../Probabilities/probabilities.h"
#include "pymatching/sparse_blossom/driver/user_graph.h" // Include necessary headers for declarations

namespace pm {
    void soft_reweight_pymatching(
        UserGraph &matching,
        const Eigen::VectorXcd& not_scaled_IQ_shot,  // Single shot, 1D array
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict, // Adjusted to hold pairs of pairs
        float p_data = -1, float p_mixed = -1, float common_measure = -1);
}


void processGraph_test(pm::UserGraph& graph);

#endif // MATCHING_H