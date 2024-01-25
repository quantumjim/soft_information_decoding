#ifndef MATCHING_H
#define MATCHING_H


#include <Eigen/Dense>  
#include "user_graph_utils.h"
#include "../Probabilities/probabilities.h"
// #include "pymatching/sparse_blossom/driver/user_graph.h" // Include necessary headers for declarations

namespace pm {

    struct ShotErrorDetails {
        std::vector<EdgeProperties> edges;
        std::vector<std::pair<int64_t, int64_t>> matched_edges;
        std::vector<int> detection_syndromes;
    };

    struct DetailedDecodeResult {
        int num_errors;
        std::vector<int> indices;
        std::vector<ShotErrorDetails> error_details;
    };

    void soft_reweight_pymatching(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        bool _resets,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict, // Adjusted to hold pairs of pairs
        float p_data = -1, float p_mixed = -1, float common_measure = -1,
        bool _adv_probs = false, bool _bimodal = false, const std::string& merge_strategy = "replace",
        float p_offset = 1.0, float p_multiplicator = 1.0, bool _ntnn_edges = false);

    void soft_reweight_1Dgauss(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        const std::map<int, std::map<std::string, float>> &gauss_params_dict);

    void soft_reweight_kde(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        bool _resets,
        const std::map<int, int> &qubit_mapping,
        std::map<int, KDE_Result> kde_dict);
        

    void reweight_edges_to_one(UserGraph &matching);

    void reweight_edges_informed(
        UserGraph &matching,  float distance, 
        float p_data, float p_mixed, float p_meas, 
        float common_measure, bool _ntnn_edges = false);

    void reweight_edges_based_on_error_probs(UserGraph &matching, const std::map<std::string, size_t>& counts, bool _resets, const std::string& method);

    ShotErrorDetails createShotErrorDetails(
        UserGraph &matching,
        std::vector<uint64_t>& detectionEvents,
        std::vector<int>& det_syndromes);

    DetailedDecodeResult decode_IQ_shots(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict, 
        float p_data, float p_mixed, float common_measure = -1,
        bool _adv_probs = false, bool _bimodal = false,
        const std::string& merge_strategy = "replace",
        bool _detailed = false,
        float p_offset = 1.0, float p_multiplicator = 1.0, bool _ntnn_edges = false);

    DetailedDecodeResult decode_IQ_1Dgauss(
        UserGraph &matching,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping, 
        const std::map<int, std::map<std::string, float>> &gauss_params_dict, 
        bool _detailed = false);
    
    DetailedDecodeResult decode_IQ_kde(
        // UserGraph &matching,
        stim::DetectorErrorModel& detector_error_model,
        const Eigen::MatrixXcd &not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int> &qubit_mapping, 
        std::map<int, KDE_Result> kde_dict,
        bool _detailed = false,
        double relError = -1, double absError = -1);
    
    DetailedDecodeResult decode_IQ_shots_flat(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict,
        bool _detailed = false);

    DetailedDecodeResult decode_IQ_shots_flat_informed(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict,
        float p_data, float p_mixed, float p_meas, float common_measure = -1,
        bool _detailed = false, bool _ntnn_edges = false);

    DetailedDecodeResult decode_IQ_shots_flat_err_probs(
        UserGraph &matching,
        int logical,
        const std::map<std::string, size_t>& counts_tot,
        bool _resets,
        const std::string& method,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict,
        bool _detailed = false);

    DetailedDecodeResult decode_IQ_shots_no_reweighting(
        UserGraph &matching,
        const Eigen::MatrixXcd& not_scaled_IQ_data,
        int synd_rounds,
        int logical,
        bool _resets,
        const std::map<int, int>& qubit_mapping,
        const std::map<int, GridData>& kde_grid_dict,
        const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict,
        bool _detailed = false);
}


void processGraph_test(pm::UserGraph& graph);

#endif // MATCHING_H