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
                                   const std::vector<double>& bandwidths, double relError, double absError, int num_points) {
    std::map<int, KDE_Result> results;

    // Checks
    if (num_points < 5) {
        throw std::invalid_argument("num_points must be at least 3 for numerical stability and !=1 for devision by 0. At 4 normalization sometimes not defined.");
    }


    double num_std_dev = 5;
    // int num_points = 51; // roughly 0.1 std dev per point
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

    #pragma omp parallel for
    for (int i = 0; i < all_memories.size(); ++i) {
        auto it = all_memories.begin();
        std::advance(it, i);
        const auto& qubit_entry = *it;

        int qubit_idx = qubit_entry.first;
        const auto& memories = qubit_entry.second;

        arma::mat combined_data = ComplexTo2DMatrix(memories.at("mmr_0"));
        combined_data.insert_cols(combined_data.n_cols, ComplexTo2DMatrix(memories.at("mmr_1"))); // Append mmr_1
        arma::vec mean, stddev;
        std::tie(mean, stddev) = StandardizeData(combined_data);

        // Fit mlpack::KDE for mmr_0 and mmr_1 separately
        arma::mat mmr_0_data = ComplexTo2DMatrix(memories.at("mmr_0"));
        StandardizeData(mmr_0_data, mean, stddev);
        arma::mat mmr_1_data = ComplexTo2DMatrix(memories.at("mmr_1"));
        StandardizeData(mmr_1_data, mean, stddev);

        // Calculate the mean and std of the mmr_0 and mmr_1 data
        arma::vec mean_mmr_0 = arma::mean(mmr_0_data, 1);
        arma::vec mean_mmr_1 = arma::mean(mmr_1_data, 1);
        arma::vec stddev_mmr_0 = arma::stddev(mmr_0_data, 0, 1);
        arma::vec stddev_mmr_1 = arma::stddev(mmr_1_data, 0, 1);

        // Split data into 80% train and 20% test
        size_t splitIndex0 = static_cast<size_t>(mmr_0_data.n_cols * 0.99); // Hardcoded 99/1 split
        size_t splitIndex1 = static_cast<size_t>(mmr_1_data.n_cols * 0.99); // Hardcoded 99/1 split

        arma::mat mmr_0_train = mmr_0_data.cols(0, splitIndex0 - 1);
        arma::mat mmr_0_test = mmr_0_data.cols(splitIndex0, mmr_0_data.n_cols - 1);
        arma::mat mmr_1_train = mmr_1_data.cols(0, splitIndex1 - 1);
        arma::mat mmr_1_test = mmr_1_data.cols(splitIndex1, mmr_1_data.n_cols - 1);

        double bestScore = -std::numeric_limits<double>::infinity();
        KDE_Result bestResult;
        
        if (relError == -1) {
            relError = mlpack::KDEDefaultParams::relError;
        }
        if (absError == -1) {
            absError = mlpack::KDEDefaultParams::absError;
        }

        mlpack::EuclideanDistance metric;

        for (double bandwidth : bandwidths) {            
            mlpack::EpanechnikovKernel epanechnikovKernel(bandwidth);          

            mlpack::KDE<mlpack::EpanechnikovKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde0(
                relError, absError, epanechnikovKernel, mlpack::KDEDefaultParams::mode, metric);

            mlpack::KDE<mlpack::EpanechnikovKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde1(
                relError, absError, epanechnikovKernel, mlpack::KDEDefaultParams::mode, metric);

            kde0.Train(mmr_0_train);
            kde1.Train(mmr_1_train);

            arma::vec all_estimates0(all_points.n_cols);
            arma::vec all_estimates1(all_points.n_cols);
            kde0.Evaluate(all_points, all_estimates0);
            kde1.Evaluate(all_points, all_estimates1);
         
            double normalization0 = arma::accu(all_estimates0) * (dx*dx); // dx*dx is the area of each grid cell
            double normalization1 = arma::accu(all_estimates1) * (dx*dx);
            
            // Evaluate on the test data and calculate a score
            arma::vec estimations0, estimations1;
            kde0.Evaluate(mmr_0_test, estimations0);
            kde1.Evaluate(mmr_1_test, estimations1);

            // Normalize the estimations
            estimations0 /= normalization0;
            estimations1 /= normalization1;

            estimations0 = arma::log(estimations0 + 1e-8);
            estimations1 = arma::log(estimations1 + 1e-8);
            double score = arma::accu(estimations0) + arma::accu(estimations1);

            if (score > bestScore) {
                bestScore = score;
                bestResult.bestBandwidth = bandwidth;
                bestResult.kde_0 = kde0;
                bestResult.kde_1 = kde1;
                bestResult.scaler_mean = mean;
                bestResult.scaler_stddev = stddev;
                bestResult.mean_mmr_0 = mean_mmr_0;
                bestResult.mean_mmr_1 = mean_mmr_1;
                bestResult.stddev_mmr_0 = stddev_mmr_0;
                bestResult.stddev_mmr_1 = stddev_mmr_1;
            }
        }
        
        if (bestResult.scaler_mean.empty() || bestResult.scaler_stddev.empty()) {
            std::cout << "!!! Error at qubit_idx: " << qubit_idx << std::endl;
            throw std::runtime_error("Mean or scaler_stddev is an empty arma::vec."); // probably because score was NaN
        }

        #pragma omp critical
        results[qubit_idx] = bestResult;
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

