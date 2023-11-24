// Maurice Hanisch mhanisc@ethz.ch
// Created 21.11.23

#include "probabilities.h"
#include <indicators/progress_bar.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <complex>


// Constructor for GridData
GridData::GridData(Eigen::MatrixXd gx, Eigen::MatrixXd gy, Eigen::MatrixXd gd0, Eigen::MatrixXd gd1)
    : grid_x(gx), grid_y(gy), grid_density_0(gd0), grid_density_1(gd1) {}


std::map<std::string, int> get_counts(
    const Eigen::MatrixXcd& not_scaled_IQ_data,
    const std::map<int, int>& qubit_mapping,
    const std::map<int, GridData>& kde_grid_dict,
    const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>>& scaler_params_dict, // Adjusted to hold pairs of pairs
    int synd_rounds) {

    std::map<std::string, int> counts;

    int distance = (not_scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

    if (not_scaled_IQ_data.cols() != (distance - 1) * synd_rounds + distance) {
        std::ostringstream error_message;
        error_message << "Number of columns in IQ data (" << not_scaled_IQ_data.cols() 
                    << ") does not match the expected value (" 
                    << ((distance - 1) * synd_rounds + distance) << ").";
        throw std::runtime_error(error_message.str());
    }

    for (int shot = 0; shot < not_scaled_IQ_data.rows(); ++shot) { 
        std::string outcome_str;
        for (int msmt = 0; msmt < not_scaled_IQ_data.cols(); ++msmt) { 

            try {
                int qubit_idx = qubit_mapping.at(msmt);
                const auto& grid_data = kde_grid_dict.at(qubit_idx);

                std::complex<double> iq_point = not_scaled_IQ_data(shot, msmt);
                const auto& [real_params, imag_params] = scaler_params_dict.at(qubit_idx);
                double real_scaled = (std::real(iq_point) - real_params.first) / real_params.second;
                double imag_scaled = (std::imag(iq_point) - imag_params.first) / imag_params.second;
                Eigen::Vector2d scaled_point = {real_scaled, imag_scaled};

                auto [outcome, density0, density1] = grid_lookup(scaled_point, grid_data);
                outcome_str += std::to_string(outcome);
            }
            catch (const std::out_of_range& e) {
                throw std::runtime_error("Qubit index " + std::to_string(msmt) + " not found in qubit mapping (qubit_mapping)");
            }

            if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
                outcome_str += " ";
            }
        }

        std::reverse(outcome_str.begin(), outcome_str.end()); // Reverse string
        counts[outcome_str]++;
    }

    return counts;
}

std::tuple<int, double, double> grid_lookup(const Eigen::Vector2d& scaled_point, const GridData& grid_data) {
    // Calculate grid spacing
    double dx = grid_data.grid_x(0, 1) - grid_data.grid_x(0, 0);
    double dy = grid_data.grid_y(1, 0) - grid_data.grid_y(0, 0);

    // Calculate indices
    int x_index = std::round((scaled_point(0) - grid_data.grid_x(0, 0)) / dx);
    int y_index = std::round((scaled_point(1) - grid_data.grid_y(0, 0)) / dy);

    // Clip indices to grid bounds
    x_index = std::clamp(x_index, 0, static_cast<int>(grid_data.grid_x.cols() - 1));
    y_index = std::clamp(y_index, 0, static_cast<int>(grid_data.grid_y.rows() - 1));

    // Retrieve densities
    double density_0 = grid_data.grid_density_0(y_index, x_index);
    double density_1 = grid_data.grid_density_1(y_index, x_index);

    // Determine outcome
    int outcome = (density_0 > density_1) ? 0 : 1;

    // Return the outcome, density_0, and density_1
    return std::make_tuple(outcome, density_0, density_1);
}


double llh_ratio(const Eigen::Vector2d& scaled_point, const GridData& grid_data) {

    auto [outcome, density0, density1] = grid_lookup(scaled_point, grid_data);


    // Implement the logic as per the Python code
    if (outcome == 0) {
        return -(density1 - density0);
    } else if (outcome == 1) {
        return -(density0 - density1);
    }  else {
        std::ostringstream error_message;
        error_message << "Invalid estimated outcome: " << outcome 
                      << ". The estimated outcome must be either 0 or 1.";
        throw std::runtime_error(error_message.str());
    }
}

// Helper function to convert NumPy array to Eigen::MatrixXd
Eigen::MatrixXd numpy_to_eigen(pybind11::array_t<double> np_array) {
    pybind11::buffer_info info = np_array.request();
    Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(static_cast<double *>(info.ptr), info.shape[0], info.shape[1]);
    return mat;
}


// PYBIND11_MODULE(cpp_soft_info, m) {
//     m.doc() = "Probabilities module"; // optional module docstring
    
//     m.def("get_counts", &get_counts, 
//           pybind11::arg("not_scaled_IQ_data"), 
//           pybind11::arg("qubit_mapping"), 
//           pybind11::arg("kde_grid_dict"), 
//           pybind11::arg("scaler_params_dict"),
//           pybind11::arg("synd_rounds"), 
//           "Get counts from not scaled IQ data");

//     m.def("numpy_to_eigen", &numpy_to_eigen, "Convert NumPy array to Eigen::MatrixXd");

//     m.def("llh_ratio", &llh_ratio, 
//           pybind11::arg("scaled_point"), 
//           pybind11::arg("grid_data"), 
//           "Calculate the log-likelihood ratio for a given point and grid data");

//     pybind11::class_<GridData>(m, "GridData")
//         .def(pybind11::init<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>())
//         .def_readwrite("grid_x", &GridData::grid_x)
//         .def_readwrite("grid_y", &GridData::grid_y)
//         .def_readwrite("grid_density_0", &GridData::grid_density_0)
//         .def_readwrite("grid_density_1", &GridData::grid_density_1);


//     // m.def("get_counts_old", &get_counts_old, 
//     //       pybind11::arg("scaled_IQ_data"), 
//     //       pybind11::arg("qubit_mapping"), 
//     //       pybind11::arg("kde_grid_dict"), 
//     //       pybind11::arg("synd_rounds"), 
//     //     //   pybind11::arg("show_progress") = false,
//     //       "Get counts from IQ data");
// }


// std::map<std::string, int> get_counts_old(const Eigen::MatrixXd& scaled_IQ_data,
//                        const std::map<int, int>& qubit_mapping,
//                        const std::map<int, GridData>& kde_grid_dict,
//                        int synd_rounds) {

//     std::map<std::string, int> counts;

//     int distance = (scaled_IQ_data.cols()/2 + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

//     if (scaled_IQ_data.cols()/2 != (distance - 1) * synd_rounds + distance) {
//         throw std::runtime_error("Number of columns in IQ data does not match the expected value");
//     }


//     for (int shot = 0; shot < scaled_IQ_data.rows(); ++shot) { 
//         std::string outcome_str;
//         for (int msmt = 0; msmt < scaled_IQ_data.cols(); msmt += 2) { 

//             try {
//                 int qubit_idx = qubit_mapping.at(msmt / 2);
//                 const auto& grid_data = kde_grid_dict.at(qubit_idx);

//                 Eigen::Vector2d scaled_point = {scaled_IQ_data(shot, msmt), scaled_IQ_data(shot, msmt + 1)};
//                 auto [outcome, density0, density1] = grid_lookup(scaled_point, grid_data);

//                 outcome_str += std::to_string(outcome);
//             }
//             catch (const std::out_of_range& e) {
//                 throw std::runtime_error("Qubit index " + std::to_string(msmt/2) + " not found in qubit mapping (qubit_mapping)");
//             }

//             if ((msmt/2 + 1) % (distance - 1) == 0 && (msmt/2 + 1) / (distance - 1) <= synd_rounds) {
//                 outcome_str += " ";
//             }
//         }

//         std::reverse(outcome_str.begin(), outcome_str.end()); // Reverse string
//         // Increment the count for the outcome string
//         counts[outcome_str]++;
//     }

//     // Sort and return the result (if necessary)
//     return counts;
// }