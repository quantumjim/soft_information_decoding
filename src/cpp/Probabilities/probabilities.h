#ifndef PROBABILITIES_H
#define PROBABILITIES_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>  // Eigen support with NumPy

#include <Eigen/Dense>  // For Eigen matrix operations
#include <vector>
#include <map>
#include <string>
#include <tuple>

// Define the GridData struct
struct GridData {
    Eigen::MatrixXd grid_x;
    Eigen::MatrixXd grid_y;
    Eigen::MatrixXd grid_density_0;
    Eigen::MatrixXd grid_density_1;

    // Constructor
    GridData(Eigen::MatrixXd gx, Eigen::MatrixXd gy, Eigen::MatrixXd gd0, Eigen::MatrixXd gd1);
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

Eigen::MatrixXd numpy_to_eigen(pybind11::array_t<double> np_array);

#endif // PROBABILITIES_H
