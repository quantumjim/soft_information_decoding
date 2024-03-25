#include "probabilities.h"
#include <chrono>
#include <iostream>
#include <limits>

#include <omp.h>


Eigen::MatrixXd quantizeMatrixVectorized(const Eigen::MatrixXd& matrix, unsigned int nBits) {
    // Fixed range
    double minVal = 0.0;
    double maxVal = 0.5;
    
    // Calculate quantization parameters
    unsigned int levels = 1 << nBits; // 2^nBits
    double quantStep = (maxVal - minVal) / (levels - 1);

    // Vectorized quantization
    // Eigen::MatrixXd normalizedMatrix = (matrix - Eigen::MatrixXd::Constant(matrix.rows(), matrix.cols(), minVal)) / (maxVal - minVal);
    Eigen::MatrixXd normalizedMatrix = matrix / maxVal;
    Eigen::MatrixXd scaledMatrix = normalizedMatrix * (levels - 1);
    Eigen::MatrixXd quantizedMatrix = (scaledMatrix.array().round()) / (levels - 1);

    // Scale back to original range
    return quantizedMatrix * maxVal;
}



std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> iqConvertor(
    const Eigen::MatrixXcd &not_scaled_IQ_data,
    const std::map<int, std::vector<int>> &inv_qubit_mapping,
    std::map<int, KDE_Result> &kde_dict,
    double relError, double absError) {

    Eigen::MatrixXd pSoftMatrix(not_scaled_IQ_data.rows(), not_scaled_IQ_data.cols());
    Eigen::MatrixXi comparisonMatrix(not_scaled_IQ_data.rows(), not_scaled_IQ_data.cols());

    double epsilon = std::numeric_limits<double>::epsilon();

    // Extracting keys from inv_qubit_mapping to a vector for OpenMP compatibility
    std::vector<int> keys;
    for (const auto& entry : inv_qubit_mapping) {
        keys.push_back(entry.first);
    }

    #pragma omp parallel for schedule(static) 
    for (size_t idx = 0; idx < keys.size(); ++idx) {
        int qubitIdx = keys[idx];
        const auto& columnIndices = inv_qubit_mapping.at(qubitIdx);
        auto &kde_entry = kde_dict.at(qubitIdx);

        // Initialize an Armadillo matrix to hold all query points for this qubit index
        arma::mat all_query_points(2, not_scaled_IQ_data.rows() * columnIndices.size());

        // Fill in all_query_points from the selected columns of not_scaled_IQ_data
        for (size_t i = 0; i < columnIndices.size(); ++i) {
            int colIndex = columnIndices[i];
            for (int row = 0; row < not_scaled_IQ_data.rows(); ++row) {
                all_query_points(0, row + i * not_scaled_IQ_data.rows()) = not_scaled_IQ_data(row, colIndex).real();
                all_query_points(1, row + i * not_scaled_IQ_data.rows()) = not_scaled_IQ_data(row, colIndex).imag();
            }
        }

        // Rescale all query points together
        all_query_points.row(0) -= kde_entry.scaler_mean[0];
        all_query_points.row(1) -= kde_entry.scaler_mean[1];
        all_query_points.row(0) /= kde_entry.scaler_stddev[0];
        all_query_points.row(1) /= kde_entry.scaler_stddev[1];

        // Prepare for KDE evaluation
        arma::vec estimations0(all_query_points.n_cols);
        arma::vec estimations1(all_query_points.n_cols);

        // Set error tolerances if specified
        if (relError != -1.0) {
            kde_entry.kde_0.RelativeError(relError);
            kde_entry.kde_1.RelativeError(relError);
        }
        if (absError != -1.0) {
            kde_entry.kde_0.AbsoluteError(absError);
            kde_entry.kde_1.AbsoluteError(absError);
        }

        // Evaluate KDE on all rescaled query points at once
        kde_entry.kde_0.Evaluate(all_query_points, estimations0);
        kde_entry.kde_1.Evaluate(all_query_points, estimations1);

        // Add small value to avoid division by zero
        estimations0 += epsilon;
        estimations1 += epsilon;
        
        for (size_t i = 0; i < columnIndices.size(); ++i) {
            int colIndex = columnIndices[i];
            for (int row = 0; row < not_scaled_IQ_data.rows(); ++row) {
                // Directly access the KDE estimation results for this point
                double estim0 = estimations0(row + i * not_scaled_IQ_data.rows());
                double estim1 = estimations1(row + i * not_scaled_IQ_data.rows());

                // Determine the smaller (p_small) and larger (p_big) of the two estimations
                double p_small, p_big;
                if (estim0 > estim1) {
                    p_small = estim1;
                    p_big = estim0;
                    comparisonMatrix(row, colIndex) = 0; // Estimation0 is greater, so assign 0
                } else {
                    p_small = estim0;
                    p_big = estim1;
                    comparisonMatrix(row, colIndex) = 1; // Estimation1 is greater or equal, so assign 1
                }

                // Calculate p_soft using the smaller (p_small) and larger (p_big) values for every element
                pSoftMatrix(row, colIndex) = 1.0 / (1.0 + p_big / p_small);
            }
        }
    }
    return std::make_tuple(pSoftMatrix, comparisonMatrix);
}


// std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> iqConvertor(
//     const Eigen::MatrixXcd &not_scaled_IQ_data,
//     const std::map<int, std::vector<int>> &inv_qubit_mapping,
//     std::map<int, KDE_Result> &kde_dict,
//     double relError, double absError) {

//     auto start_overall = std::chrono::high_resolution_clock::now();

//     Eigen::MatrixXd pSoftMatrix(not_scaled_IQ_data.rows(), not_scaled_IQ_data.cols());
//     Eigen::MatrixXi comparisonMatrix(not_scaled_IQ_data.rows(), not_scaled_IQ_data.cols());
//     double epsilon = std::numeric_limits<double>::epsilon();

//     for (const auto &entry : inv_qubit_mapping) {
//         const auto &qubitIdx = entry.first;
//         const auto &columnIndices = entry.second;
//         auto &kde_entry = kde_dict.at(qubitIdx);

//         auto start_step = std::chrono::high_resolution_clock::now();

//         arma::mat all_query_points(2, not_scaled_IQ_data.rows() * columnIndices.size());
//         for (size_t i = 0; i < columnIndices.size(); ++i) {
//             int colIndex = columnIndices[i];
//             for (int row = 0; row < not_scaled_IQ_data.rows(); ++row) {
//                 all_query_points(0, row + i * not_scaled_IQ_data.rows()) = not_scaled_IQ_data(row, colIndex).real();
//                 all_query_points(1, row + i * not_scaled_IQ_data.rows()) = not_scaled_IQ_data(row, colIndex).imag();
//             }
//         }

//         auto end_step = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsed = end_step - start_step;
//         std::cout << "Time to fill query points: " << elapsed.count() << " s\n";

//         start_step = std::chrono::high_resolution_clock::now();

//         all_query_points.row(0) -= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_mean[0]);
//         all_query_points.row(1) -= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_mean[1]);
//         all_query_points.row(0) /= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_stddev[0]);
//         all_query_points.row(1) /= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_stddev[1]);

//         end_step = std::chrono::high_resolution_clock::now();
//         elapsed = end_step - start_step;
//         std::cout << "Time to rescale query points: " << elapsed.count() << " s\n";

//         start_step = std::chrono::high_resolution_clock::now();

//         arma::vec estimations0(all_query_points.n_cols);
//         arma::vec estimations1(all_query_points.n_cols);

//         if (relError != -1.0) {
//             kde_entry.kde_0.RelativeError(relError);
//             kde_entry.kde_1.RelativeError(relError);
//         }
//         if (absError != -1.0) {
//             kde_entry.kde_0.AbsoluteError(absError);
//             kde_entry.kde_1.AbsoluteError(absError);
//         }

//         kde_entry.kde_0.Evaluate(all_query_points, estimations0);
//         kde_entry.kde_1.Evaluate(all_query_points, estimations1);

//         end_step = std::chrono::high_resolution_clock::now();
//         elapsed = end_step - start_step;
//         std::cout << "Time for KDE evaluation: " << elapsed.count() << " s\n";

//         start_step = std::chrono::high_resolution_clock::now();

//         estimations0 += epsilon;
//         estimations1 += epsilon;

//         end_step = std::chrono::high_resolution_clock::now();
//         elapsed = end_step - start_step;
//         std::cout << "Time to add epsilon: " << elapsed.count() << " s\n";

//         start_step = std::chrono::high_resolution_clock::now();

//         for (size_t i = 0; i < columnIndices.size(); ++i) {
//             int colIndex = columnIndices[i];
//             for (int row = 0; row < not_scaled_IQ_data.rows(); ++row) {
//                 double estim0 = estimations0(row + i * not_scaled_IQ_data.rows());
//                 double estim1 = estimations1(row + i * not_scaled_IQ_data.rows());

//                 double p_small, p_big;
//                 if (estim0 > estim1) {
//                     p_small = estim1;
//                     p_big = estim0;
//                     comparisonMatrix(row, colIndex) = 0; // Estimation0 is greater, so assign 0
//                 } else {
//                     p_small = estim0;
//                     p_big = estim1;
//                     comparisonMatrix(row, colIndex) = 1; // Estimation1 is greater or equal, so assign 1
//                 }
//                             // Calculate p_soft using the smaller (p_small) and larger (p_big) values for every element
//                 pSoftMatrix(row, colIndex) = 1.0 / (1.0 + p_big / p_small);
//             }
//         }

//         end_step = std::chrono::high_resolution_clock::now();
//         elapsed = end_step - start_step;
//         std::cout << "Time to process results and populate matrices: " << elapsed.count() << " s\n";
//     }

//     auto end_overall = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_overall = end_overall - start_overall;
//     std::cout << "Total execution time for iqConvertor: " << elapsed_overall.count() << " s\n";

//     return std::make_tuple(pSoftMatrix, comparisonMatrix);
// }

