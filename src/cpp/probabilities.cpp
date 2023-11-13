#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>  // Assuming we are using Eigen for matrix operations

struct GridData {
    Eigen::MatrixXd grid_x;
    Eigen::MatrixXd grid_y;
    Eigen::MatrixXd grid_density_0;
    Eigen::MatrixXd grid_density_1;
};


// Forward declaration of grid_lookup
int grid_lookup(const Eigen::Vector2d& point, const GridData& grid_data);



std::map<std::string, int> get_counts(const Eigen::MatrixXd& scaled_IQ_data,
                       const std::map<int, int>& qubit_mapping,
                       const std::map<int, GridData>& kde_grid_dict,
                       int synd_rounds) {

    std::map<std::string, int> counts;

    int distance = (scaled_IQ_data.cols() + synd_rounds) / (synd_rounds + 1); // Hardcoded for RepCodes

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
                std::cerr << "Error: Qubit index not found in mapping or grid data: " << e.what() << '\n';
                outcome_str += "?";  
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


PYBIND11_MODULE(cpp_probabilities, m) {
    m.doc() = "Probabilities module"; // optional module docstring
    m.def("get_counts", &get_counts, "Soft_info get_counts function");
}
