// Maurice Hanisch mhanisc@ethz.ch
// Created 16.01.24

#include "probabilities.h"
#include <cstring> // for std::memcpy
#include <optional>

#include <chrono>



arma::mat ComplexTo2DMatrix(const std::vector<std::complex<double>>& complexVec) {
    arma::mat mat(2, complexVec.size());
    for (size_t i = 0; i < complexVec.size(); ++i) {
        mat(0, i) = complexVec[i].real();
        mat(1, i) = complexVec[i].imag();
    }
    return mat;
}


std::tuple<arma::vec, arma::vec> StandardizeData(arma::mat& data, 
                                                 std::optional<arma::vec> mean, 
                                                 std::optional<arma::vec> stddev) 
{
    if (!mean.has_value()) {
        mean = arma::mean(data, 1);
    }
    if (!stddev.has_value()) {
        stddev = arma::stddev(data, 0, 1); // Normalizing by N, not N-1
    }

    for (size_t i = 0; i < data.n_rows; ++i) {
        for (size_t j = 0; j < data.n_cols; ++j) {
            data(i, j) = (data(i, j) - mean.value()[i]) / stddev.value()[i];
        }
    }

    return std::make_tuple(mean.value(), stddev.value());
}


std::map<int, KDE_Result> get_KDEs(const std::map<int, std::map<std::string, std::vector<std::complex<double>>>>& all_memories,
                                   const std::vector<double>& bandwidths, double relError, double absError) {
    std::map<int, KDE_Result> results;

    for (const auto& qubit_entry : all_memories) {
        int qubit_idx = qubit_entry.first;
        const auto& memories = qubit_entry.second;

        // // Combine and scale data
        // std::cout << "mmr_0 size: " << memories.at("mmr_0").size() << std::endl;
        // std::cout << "mmr_1 size: " << memories.at("mmr_1").size() << std::endl;
        arma::mat combined_data = ComplexTo2DMatrix(memories.at("mmr_0"));
        // std::cout << "combined_data size: " << combined_data.n_cols << std::endl;
        combined_data.insert_cols(combined_data.n_cols, ComplexTo2DMatrix(memories.at("mmr_1"))); // Append mmr_1
        // std::cout << "combined_data size: " << combined_data.n_cols << std::endl;
        arma::vec mean, stddev;
        std::tie(mean, stddev) = StandardizeData(combined_data);
        // std::cout << "mean and std" << mean << " " << stddev << std::endl;

        // Fit mlpack::KDE for mmr_0 and mmr_1 separately
        arma::mat mmr_0_data = ComplexTo2DMatrix(memories.at("mmr_0"));
        StandardizeData(mmr_0_data, mean, stddev);
        // std::cout << "mmr_0_data size after standardization: " << mmr_0_data.n_cols << std::endl;
        arma::mat mmr_1_data = ComplexTo2DMatrix(memories.at("mmr_1"));
        StandardizeData(mmr_1_data, mean, stddev);
        // std::cout << "mmr_1_data size before standardization: " << mmr_1_data.n_cols << std::endl;
        
        // throw std::runtime_error("KDE.cpp: get_KDEs: mmr_1_data size before standardization: " + std::to_string(mmr_1_data.n_cols));
        double bandwidth = bandwidths[qubit_idx];
        // mlpack::KDE<mlpack::GaussianKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde0(bandwidth);
        // mlpack::KDE<mlpack::GaussianKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde1(bandwidth);

        if (relError == -1) {
            relError = mlpack::KDEDefaultParams::relError;
        }
        if (absError == -1) {
            absError = mlpack::KDEDefaultParams::absError;
        }

        // mlpack::GaussianKernel kernel(bandwidth);
        mlpack::EpanechnikovKernel epanechnikovKernel(bandwidth);
        mlpack::EuclideanDistance metric;

        mlpack::KDE<mlpack::EpanechnikovKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde0(
            relError, absError, epanechnikovKernel, mlpack::KDEDefaultParams::mode, metric);

        mlpack::KDE<mlpack::EpanechnikovKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde1(
            relError, absError, epanechnikovKernel, mlpack::KDEDefaultParams::mode, metric);

        kde0.Train(mmr_0_data);
        kde1.Train(mmr_1_data);


        // Save the results
        KDE_Result result;
        result.kde_0 = kde0;
        result.kde_1 = kde1;
        result.scaler_mean = mean;
        result.scaler_stddev = stddev;

        results[qubit_idx] = result;
    }
    
    return results;
}


Eigen::VectorXd armaVecToEigenVec(const arma::vec& armaVec) {
    Eigen::VectorXd eigenVec(armaVec.n_elem);
    std::memcpy(eigenVec.data(), armaVec.memptr(), armaVec.n_elem * sizeof(double));
    return eigenVec;
}




std::map<int, std::tuple<Eigen::VectorXd, Eigen::VectorXd>> GenerateGridAndEstimateDensity(
    std::map<int, KDE_Result> kde_dict, int num_points, double num_std_dev) {

    auto start = std::chrono::high_resolution_clock::now();
    std::map<int, std::tuple<Eigen::VectorXd, Eigen::VectorXd>> grid_data_map;

    double dx = 2 * num_std_dev / (num_points - 1);

    arma::mat all_points(2, num_points * num_points);
    int idx = 0;
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            all_points(0, idx) = -num_std_dev + i * dx; // x-coordinate
            all_points(1, idx) = -num_std_dev + j * dx; // y-coordinate
            ++idx;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Grid generation time: " << elapsed.count() << " seconds" << std::endl;

    for (auto& qubit_entry : kde_dict) {
        start = std::chrono::high_resolution_clock::now();

        int qubit_idx = qubit_entry.first;
        auto& kde0 = qubit_entry.second.kde_0; // Assuming you have access to KDE models like this
        auto& kde1 = qubit_entry.second.kde_1;

        arma::vec estimations0(all_points.n_cols);
        arma::vec estimations1(all_points.n_cols);

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "KDE initialization time for qubit " << qubit_idx << ": " << elapsed.count() << " seconds" << std::endl;

        start = std::chrono::high_resolution_clock::now();

        kde0.Evaluate(all_points, estimations0);
        kde1.Evaluate(all_points, estimations1);

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "KDE evaluation time for qubit " << qubit_idx << ": " << elapsed.count() << " seconds" << std::endl;

        start = std::chrono::high_resolution_clock::now();

        Eigen::VectorXd estimations0_eigen = armaVecToEigenVec(estimations0);
        Eigen::VectorXd estimations1_eigen = armaVecToEigenVec(estimations1);

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Conversion to Eigen time for qubit " << qubit_idx << ": " << elapsed.count() << " seconds" << std::endl;

        grid_data_map[qubit_idx] = std::make_tuple(estimations0_eigen, estimations1_eigen);
    }

    return grid_data_map;
}


// std::map<int, std::tuple<Eigen::VectorXd, Eigen::VectorXd>> GenerateGridAndEstimateDensity(std::map<int, KDE_Result> kde_dict, 
//                                                        int num_points, double num_std_dev) {
//     std::map<int, std::tuple<Eigen::VectorXd, Eigen::VectorXd>> grid_data_map;

//     double dx = 2 * num_std_dev / (num_points - 1);

//     arma::mat all_points(2, num_points * num_points);
//     int idx = 0;
//     for (int i = 0; i < num_points; ++i) {
//         for (int j = 0; j < num_points; ++j) {
//             all_points(0, idx) = -num_std_dev + i * dx; // x-coordinate
//             all_points(1, idx) = -num_std_dev + j * dx; // y-coordinate
//             ++idx;
//         }
//     }
//     for (auto& qubit_entry : kde_dict) {
//         int qubit_idx = qubit_entry.first;
//         auto& kde0 = qubit_entry.second.kde_0; // Assuming you have access to KDE models like this
//         auto& kde1 = qubit_entry.second.kde_1;

//         arma::vec estimations0(all_points.n_cols);
//         arma::vec estimations1(all_points.n_cols);
//         kde0.Evaluate(all_points, estimations0);
//         kde1.Evaluate(all_points, estimations1);      

//         Eigen::VectorXd estimations0_eigen = armaVecToEigenVec(estimations0);
//         Eigen::VectorXd estimations1_eigen = armaVecToEigenVec(estimations1);

//         grid_data_map[qubit_idx] = std::make_tuple(estimations0_eigen, estimations1_eigen);
//     }

//     return grid_data_map;
// }

