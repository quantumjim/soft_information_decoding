#ifndef PROBABILITIES_H
#define PROBABILITIES_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>  // Eigen support with NumPy

#include <Eigen/Dense>  // For Eigen matrix operations
#include <vector>
#include <complex>
#include <map>
#include <string>
#include <tuple>
#include <stdexcept>

#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <armadillo>


// Define the GridData struct
struct GridData {
    Eigen::MatrixXd grid_x;
    Eigen::MatrixXd grid_y;
    Eigen::MatrixXd grid_density_0;
    Eigen::MatrixXd grid_density_1;

    // Constructor
    GridData(Eigen::MatrixXd gx, Eigen::MatrixXd gy, Eigen::MatrixXd gd0, Eigen::MatrixXd gd1);
};

// Define the KDE_Result struct because imported in KDE module
struct KDE_Result {
    mlpack::KDE<mlpack::EpanechnikovKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde_0;
    mlpack::KDE<mlpack::EpanechnikovKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde_1;
    double bestBandwidth;
    arma::vec scaler_mean;
    arma::vec scaler_stddev;
};




// Function declarations
std::tuple<int, double, double> grid_lookup(const Eigen::Vector2d& scaled_point, const GridData& grid_data);

std::map<std::string, int> get_counts(
    const Eigen::MatrixXcd& not_scaled_IQ_data,
    const std::map<int, int>& qubit_mapping,
    const std::map<int, GridData>& kde_grid_dict,
    const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict,
    int synd_rounds);

double llh_ratio(const Eigen::Vector2d& scaled_point, const GridData& grid_data, double bimodal_prob = -1);

std::map<std::string, float> llh_ratio_1Dgauss(
    double rpoint, std::map<std::string,float> gauss_params);

std::map<std::string, float> llh_ratio_kde(std::complex<double> not_scaled_point, KDE_Result kde_entry);

std::map<std::string, int> get_counts_1Dgauss(
    const Eigen::MatrixXcd& not_scaled_IQ_data,
    const std::map<int, int>& qubit_mapping,
    const std::map<int, std::map<std::string, float>> &gauss_params_dict, 
    int synd_rounds);

std::map<std::string, int> get_counts_kde(
    const Eigen::MatrixXcd& not_scaled_IQ_data,
    const std::map<int, int>& qubit_mapping,
    std::map<int, KDE_Result> kde_dict, 
    int synd_rounds);

Eigen::MatrixXd numpy_to_eigen(pybind11::array_t<double> np_array);



////////// KDE //////////

arma::mat ComplexTo2DMatrix(const std::vector<std::complex<double>>& complexVec);

std::tuple<arma::vec, arma::vec> StandardizeData(arma::mat& data, 
                                                 std::optional<arma::vec> mean = std::nullopt, 
                                                 std::optional<arma::vec> stddev = std::nullopt);

std::map<int, KDE_Result> get_KDEs(const std::map<int, std::map<std::string, std::vector<std::complex<double>>>>& all_memories,
                                   const std::vector<double>& bandwidths, double relError = -1, double absError = -1, int num_points = 51);

std::map<int, std::tuple<Eigen::VectorXd, Eigen::VectorXd>> GenerateGridAndEstimateDensity(std::map<int, KDE_Result> kde_dict, 
                                                       int num_points, double num_std_dev);


#endif // PROBABILITIES_H
