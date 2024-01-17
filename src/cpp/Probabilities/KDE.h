#ifndef KDE_H
#define KDE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>  // Eigen support with NumPy

#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <armadillo>

#include <complex>
#include <vector>
#include <map>
#include <string>
#include <tuple>

#include "probabilities.h"

struct KDE_Result {
    mlpack::KDE<mlpack::GaussianKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde_0;
    mlpack::KDE<mlpack::GaussianKernel, mlpack::EuclideanDistance, arma::mat, mlpack::KDTree> kde_1;
    arma::vec scaler_mean;
    arma::vec scaler_stddev;
};

arma::mat ComplexTo2DMatrix(const std::vector<std::complex<double>>& complexVec);

std::tuple<arma::vec, arma::vec> StandardizeData(arma::mat& data, 
                                                 std::optional<arma::vec> mean = std::nullopt, 
                                                 std::optional<arma::vec> stddev = std::nullopt);

std::map<int, KDE_Result> get_KDEs(const std::map<int, std::map<std::string, std::vector<std::complex<double>>>>& all_memories,
                                   const std::vector<double>& bandwidths);

std::map<int, std::tuple<Eigen::VectorXd, Eigen::VectorXd>> GenerateGridAndEstimateDensity(std::map<int, KDE_Result> kde_dict, 
                                                       int num_points, double num_std_dev);

#endif // KDE_H
