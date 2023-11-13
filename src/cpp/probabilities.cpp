#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>  // Add this include for Eigen support with NumPy

#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>

#include <indicators/progress_bar.hpp>
#include <Eigen/Dense>  // Assuming we are using Eigen for matrix operations

struct GridData {
    Eigen::MatrixXd grid_x;
    Eigen::MatrixXd grid_y;
    Eigen::MatrixXd grid_density_0;
    Eigen::MatrixXd grid_density_1;
    // Constructor, if needed
    GridData(Eigen::MatrixXd gx, Eigen::MatrixXd gy, Eigen::MatrixXd gd0, Eigen::MatrixXd gd1)
        : grid_x(gx), grid_y(gy), grid_density_0(gd0), grid_density_1(gd1) {}
};



// Forward declaration of grid_lookup
int grid_lookup(const Eigen::Vector2d& point, const GridData& grid_data);



std::map<std::string, int> get_counts(const Eigen::MatrixXd& scaled_IQ_data,
                       const std::map<int, int>& qubit_mapping,
                       const std::map<int, GridData>& kde_grid_dict,
                       int synd_rounds) {

    std::map<std::string, int> counts;

    int distance = (scaled_IQ_data.cols()/2 + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

    if (scaled_IQ_data.cols()/2 != (distance - 1) * synd_rounds + distance) {
        throw std::runtime_error("Number of columns in IQ data does not match the expected value");
    }


    for (int shot = 0; shot < scaled_IQ_data.rows(); ++shot) { 
        std::string outcome_str;
        for (int msmt = 0; msmt < scaled_IQ_data.cols(); msmt += 2) { 

            try {
                int qubit_idx = qubit_mapping.at(msmt / 2);
                const auto& grid_data = kde_grid_dict.at(qubit_idx);

                Eigen::Vector2d scaled_point = {scaled_IQ_data(shot, msmt), scaled_IQ_data(shot, msmt + 1)};
                int outcome = grid_lookup(scaled_point, grid_data);

                outcome_str += std::to_string(outcome);
            }
            catch (const std::out_of_range& e) {
                throw std::runtime_error("Qubit index " + std::to_string(msmt/2) + " not found in qubit mapping (qubit_mapping)");
            }

            if ((msmt/2 + 1) % (distance - 1) == 0 && (msmt/2 + 1) / (distance - 1) <= synd_rounds) {
                outcome_str += " ";
            }
        }

        std::reverse(outcome_str.begin(), outcome_str.end()); // Reverse string
        // Increment the count for the outcome string
        counts[outcome_str]++;
    }

    // Sort and return the result (if necessary)
    return counts;
}

int grid_lookup(const Eigen::Vector2d& point, const GridData& grid_data) {
    // Calculate grid spacing
    double dx = grid_data.grid_x(0, 1) - grid_data.grid_x(0, 0);
    double dy = grid_data.grid_y(1, 0) - grid_data.grid_y(0, 0);

    // Calculate indices
    int x_index = std::round((point(0) - grid_data.grid_x(0, 0)) / dx);
    int y_index = std::round((point(1) - grid_data.grid_y(0, 0)) / dy);

    // Clip indices to grid bounds
    x_index = std::clamp(x_index, 0, static_cast<int>(grid_data.grid_x.cols() - 1));
    y_index = std::clamp(y_index, 0, static_cast<int>(grid_data.grid_y.rows() - 1));

    // Retrieve densities
    double density_0 = grid_data.grid_density_0(y_index, x_index);
    double density_1 = grid_data.grid_density_1(y_index, x_index);

    // Determine outcome
    return (density_0 > density_1) ? 0 : 1;
}


// Helper function to convert NumPy array to Eigen::MatrixXd
Eigen::MatrixXd numpy_to_eigen(pybind11::array_t<double> np_array) {
    pybind11::buffer_info info = np_array.request();
    Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(static_cast<double *>(info.ptr), info.shape[0], info.shape[1]);
    return mat;
}


PYBIND11_MODULE(cpp_probabilities, m) {
    m.doc() = "Probabilities module"; // optional module docstring

    m.def("get_counts", &get_counts, 
          pybind11::arg("scaled_IQ_data"), 
          pybind11::arg("qubit_mapping"), 
          pybind11::arg("kde_grid_dict"), 
          pybind11::arg("synd_rounds"), 
        //   pybind11::arg("show_progress") = false,
          "Get counts from IQ data");

    m.def("numpy_to_eigen", &numpy_to_eigen, "Convert NumPy array to Eigen::MatrixXd");

    pybind11::class_<GridData>(m, "GridData")
        .def(pybind11::init<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>())
        .def_readwrite("grid_x", &GridData::grid_x)
        .def_readwrite("grid_y", &GridData::grid_y)
        .def_readwrite("grid_density_0", &GridData::grid_density_0)
        .def_readwrite("grid_density_1", &GridData::grid_density_1);
}
