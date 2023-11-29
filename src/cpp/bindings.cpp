#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "Probabilities/probabilities.h"  // Include probabilities header
#include "PyMatching/matching_graph.h"  // Include matching header
#include "PyMatching/user_graph_utils.h"  // Include user graph utils header

#include "pymatching/sparse_blossom/driver/user_graph.h" // Include necessary headers for declarations

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

    // m.def("print_edges_of_graph", &print_edges_of_graph, "Print the edges of a matching graph");
    m.def("processGraph_test", &processGraph_test, "Function to test process UserGraph");

    py::class_<GridData>(m, "GridData")
        .def(py::init<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>())
        .def_readwrite("grid_x", &GridData::grid_x)
        .def_readwrite("grid_y", &GridData::grid_y)
        .def_readwrite("grid_density_0", &GridData::grid_density_0)
        .def_readwrite("grid_density_1", &GridData::grid_density_1);   
    

    // user_graph_utils.h bindings
    m.def("get_edges", [](const pm::UserGraph& graph) {
        auto edges = get_edges(graph);
        py::list py_edges;
        for (const auto& edge : edges) {
            py::dict attrs;
            attrs["fault_ids"] = py::cast(edge.attributes.fault_ids);
            attrs["weight"] = edge.attributes.weight;
            attrs["error_probability"] = edge.attributes.error_probability;

            if (edge.node2 == SIZE_MAX) {
                py_edges.append(py::make_tuple(edge.node1, py::none(), attrs));
            } else {
                py_edges.append(py::make_tuple(edge.node1, edge.node2, attrs));
            }
        }
        return py_edges;
    }, "Get edges of a matching graph");

    m.def("add_edge", &pm::add_edge, 
          "Add or merge an edge to the user graph",
          py::arg("graph"), py::arg("node1"), py::arg("node2"), 
          py::arg("observables"), py::arg("weight"), 
          py::arg("error_probability"), py::arg("merge_strategy"));
    
    m.def("add_boundary_edge", &pm::add_boundary_edge, 
      "Add or merge a boundary edge to the user graph",
      py::arg("graph"), py::arg("node"), py::arg("observables"), 
      py::arg("weight"), py::arg("error_probability"), py::arg("merge_strategy"));
    
    m.def("decode", &pm::decode, 
      "Decode a matching graph",
      py::arg("graph"), py::arg("detection_events"));

    m.def("counts_to_det_syndr", &counts_to_det_syndr, 
      "Convert counts to deterministic syndromes",
      py::arg("input_str"), py::arg("_resets") = false, 
      py::arg("verbose") = false);

    m.def("syndromeArrayToDetectionEvents", &syndromeArrayToDetectionEvents, 
      "Convert syndrome array to detection events",
      py::arg("z"), py::arg("num_detectors"), py::arg("boundary_length"));
    
    m.def("soft_reweight_pymatching", &pm::soft_reweight_pymatching, 
      "Reweight a matching graph using soft information",
      py::arg("matching"), py::arg("not_scaled_IQ_data"), 
      py::arg("synd_rounds"), py::arg("qubit_mapping"), 
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"), 
      py::arg("p_data"), py::arg("p_mixed"), py::arg("common_measure"));
    
    m.def("reweight_edges_to_one", &pm::reweight_edges_to_one, 
      "Reweight a matching graph to have edge weights of 1",
      py::arg("matching"));

    m.def("reweight_edges_to_one_diag", &pm::reweight_edges_to_one_diag, 
      "Reweight a matching graph to have edge weights of 1 and use diagonal edges",
      py::arg("matching"), py::arg("p_mixed"), py::arg("distance"));
    
    m.def("decode_IQ_shots", &pm::decode_IQ_shots, 
      "Decode a matching graph using IQ data",
      py::arg("matching"), py::arg("not_scaled_IQ_data"), 
      py::arg("synd_rounds"), py::arg("qubit_mapping"), 
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"), 
      py::arg("p_data"), py::arg("p_mixed"), py::arg("common_measure"));

    m.def("decode_IQ_shots_flat", &pm::decode_IQ_shots_flat,
      "Decode a matching graph using IQ data but weight edges to 1",
      py::arg("matching"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("qubit_mapping"),
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"));
    
    m.def("decode_IQ_shots_flat_diag", &pm::decode_IQ_shots_flat_diag,
      "Decode a matching graph using IQ data but weight edges to 1 and use diagonal edges",
      py::arg("matching"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("qubit_mapping"),
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"),
      py::arg("p_mixed"));
}
