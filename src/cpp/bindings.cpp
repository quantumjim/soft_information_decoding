#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "Probabilities/probabilities.h"  // Include your probabilities header
#include "PyMatching/matching_graph.h"  // Include your matching header

namespace py = pybind11;

PYBIND11_MODULE(cpp_soft_info, m) {
    m.doc() = "Probabilities module"; // Optional module docstring
    
    m.def("get_counts", &get_counts, 
          py::arg("not_scaled_IQ_data"), 
          py::arg("qubit_mapping"), 
          py::arg("kde_grid_dict"), 
          py::arg("scaler_params_dict"),
          py::arg("synd_rounds"), 
          "Get counts from not scaled IQ data");

    m.def("numpy_to_eigen", &numpy_to_eigen, "Convert NumPy array to Eigen::MatrixXd");

    m.def("llh_ratio", &llh_ratio, 
          py::arg("scaled_point"), 
          py::arg("grid_data"), 
          "Calculate the log-likelihood ratio for a given point and grid data");

    m.def("processGraph_test", &processGraph_test, "Function to process UserGraph");

    py::class_<GridData>(m, "GridData")
        .def(py::init<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>())
        .def_readwrite("grid_x", &GridData::grid_x)
        .def_readwrite("grid_y", &GridData::grid_y)
        .def_readwrite("grid_density_0", &GridData::grid_density_0)
        .def_readwrite("grid_density_1", &GridData::grid_density_1);
}
