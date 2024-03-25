#include "probabilities.h"
#include <chrono>
#include <iostream>




std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> iqConvertor(
    const Eigen::MatrixXcd &not_scaled_IQ_data,
    const std::map<int, std::vector<int>> &inv_qubit_mapping,
    std::map<int, KDE_Result> &kde_dict,
    double relError, double absError) {

    Eigen::MatrixXd pSoftMatrix(not_scaled_IQ_data.rows(), not_scaled_IQ_data.cols());
    Eigen::MatrixXi comparisonMatrix(not_scaled_IQ_data.rows(), not_scaled_IQ_data.cols());

    for (const auto &entry : inv_qubit_mapping) {
        const auto &qubitIdx = entry.first;
        const auto &columnIndices = entry.second;
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
        all_query_points.row(0) -= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_mean[0]);
        all_query_points.row(1) -= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_mean[1]);
        all_query_points.row(0) /= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_stddev[0]);
        all_query_points.row(1) /= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_stddev[1]);


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

//     auto start_time = std::chrono::high_resolution_clock::now();

//     Eigen::MatrixXd pSoftMatrix(not_scaled_IQ_data.rows(), not_scaled_IQ_data.cols());
//     Eigen::MatrixXi comparisonMatrix(not_scaled_IQ_data.rows(), not_scaled_IQ_data.cols());

//     for (const auto &entry : inv_qubit_mapping) {
//         auto operation_start = std::chrono::high_resolution_clock::now();

//         const auto &qubitIdx = entry.first;
//         const auto &columnIndices = entry.second;
//         auto &kde_entry = kde_dict.at(qubitIdx);

//         auto populate_start = std::chrono::high_resolution_clock::now();
//         arma::mat all_query_points(2, not_scaled_IQ_data.rows() * columnIndices.size());
//         for (size_t i = 0; i < columnIndices.size(); ++i) {
//             int colIndex = columnIndices[i];
//             for (int row = 0; row < not_scaled_IQ_data.rows(); ++row) {
//                 all_query_points(0, row + i * not_scaled_IQ_data.rows()) = not_scaled_IQ_data(row, colIndex).real();
//                 all_query_points(1, row + i * not_scaled_IQ_data.rows()) = not_scaled_IQ_data(row, colIndex).imag();
//             }
//         }
//         auto populate_end = std::chrono::high_resolution_clock::now();

//         auto rescale_start = std::chrono::high_resolution_clock::now();
//         all_query_points.row(0) -= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_mean[0]);
//         all_query_points.row(1) -= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_mean[1]);
//         all_query_points.row(0) /= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_stddev[0]);
//         all_query_points.row(1) /= arma::rowvec(all_query_points.n_cols).fill(kde_entry.scaler_stddev[1]);
//         auto rescale_end = std::chrono::high_resolution_clock::now();

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

//         auto kde_start = std::chrono::high_resolution_clock::now();
//         kde_entry.kde_0.Evaluate(all_query_points, estimations0);
//         kde_entry.kde_1.Evaluate(all_query_points, estimations1);
//         auto kde_end = std::chrono::high_resolution_clock::now();

//         auto matrix_population_start = std::chrono::high_resolution_clock::now();
//         for (size_t i = 0; i < columnIndices.size(); ++i) {
//             int colIndex = columnIndices[i];
//             for (int row = 0; row < not_scaled_IQ_data.rows(); ++row) {
//                 double estim0 = estimations0(row + i * not_scaled_IQ_data.rows());
//                 double estim1 = estimations1(row + i * not_scaled_IQ_data.rows());
//                 double p_small, p_big;
//                 if (estim0 > estim1) {
//                     p_small = estim1;
//                     p_big = estim0;
//                     comparisonMatrix(row, colIndex) = 0;
//                 } else {
//                     p_small = estim0;
//                     p_big = estim1;
//                     comparisonMatrix(row, colIndex) = 1;
//                 }
//                 pSoftMatrix(row, colIndex) = 1.0 / (1.0 + p_big / p_small);
//             }
//         }
//         auto matrix_population_end = std::chrono::high_resolution_clock::now();

//         auto operation_end = std::chrono::high_resolution_clock::now();
//         std::cout << "Operation for qubit " << qubitIdx << " took "
//                   << std::chrono::duration_cast<std::chrono::milliseconds>(operation_end - operation_start).count()
//                   << " milliseconds." << std::endl;
//         std::cout << "    Rescaling took "
//                   << std::chrono::duration_cast<std::chrono::milliseconds>(rescale_end - rescale_start).count()
//                   << " milliseconds." << std::endl;
//         std::cout << "    Populating query points took "
//                     << std::chrono::duration_cast<std::chrono::milliseconds>(populate_end - populate_start).count()
//                     << " milliseconds." << std::endl;
//         std::cout << "    Matrix population took "
//                     << std::chrono::duration_cast<std::chrono::milliseconds>(matrix_population_end - matrix_population_start).count()
//                     << " milliseconds." << std::endl;
//         std::cout << "    KDE evaluation took "
//                   << std::chrono::duration_cast<std::chrono::milliseconds>(kde_end - kde_start).count()
//                   << " milliseconds." << std::endl;
//     }

//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::cout << "Total function execution time: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
//               << " milliseconds." << std::endl;

//     return std::make_tuple(pSoftMatrix, comparisonMatrix);
// }

