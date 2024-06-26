#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "Probabilities/probabilities.h"  
#include "PyMatching/matching_graph.h"  
#include "PyMatching/user_graph_utils.h"  
#include "PyMatching/predecoders.h"  
#include "PyMatching/convdecoders.h"
#include "StimUtils/StimUtils.h"

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
          py::arg("bimodal_prob") = -1,
          "Calculate the log-likelihood ratio for a given point and grid data");

    // m.def("print_edges_of_graph", &print_edges_of_graph, "Print the edges of a matching graph");
    m.def("processGraph_test", &processGraph_test, "Function to test process UserGraph");



    //////////// STRUCTS ////////////

    py::class_<GridData>(m, "GridData")
        .def(py::init<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>())
        .def_readwrite("grid_x", &GridData::grid_x)
        .def_readwrite("grid_y", &GridData::grid_y)
        .def_readwrite("grid_density_0", &GridData::grid_density_0)
        .def_readwrite("grid_density_1", &GridData::grid_density_1);   
        
    py::class_<EdgeAttributes>(m, "EdgeAttributes")
        .def(py::init<>()) // Default constructor
        .def_readwrite("fault_ids", &EdgeAttributes::fault_ids)
        .def_readwrite("weight", &EdgeAttributes::weight)
        .def_readwrite("error_probability", &EdgeAttributes::error_probability);

    py::class_<EdgeProperties>(m, "EdgeProperties")
        .def(py::init<>())
        .def_readwrite("node1", &EdgeProperties::node1)
        .def_readwrite("node2", &EdgeProperties::node2)
        .def_readwrite("attributes", &EdgeProperties::attributes);

    py::class_<pm::ShotErrorDetails>(m, "ShotErrorDetails")
        .def(py::init<>())
        .def_readwrite("edges", &pm::ShotErrorDetails::edges)
        .def_readwrite("matched_edges", &pm::ShotErrorDetails::matched_edges)
        .def_readwrite("detection_syndromes", &pm::ShotErrorDetails::detection_syndromes);

    py::class_<pm::DetailedDecodeResult>(m, "DetailedDecodeResult")
        .def(py::init<>())
        .def_readwrite("num_errors", &pm::DetailedDecodeResult::num_errors)
        .def_readwrite("indices", &pm::DetailedDecodeResult::indices)
        .def_readwrite("error_details", &pm::DetailedDecodeResult::error_details);

    py::class_<KDE_Result>(m, "KDE_Result")
        .def(py::init<>())
        .def_readwrite("kde_0", &KDE_Result::kde_0)
        .def_readwrite("kde_1", &KDE_Result::kde_1)
        .def_readwrite("bestBandwidth", &KDE_Result::bestBandwidth)
        .def_readwrite("scaler_mean", &KDE_Result::scaler_mean)
        .def_readwrite("scaler_stddev", &KDE_Result::scaler_stddev)
        .def_readwrite("mean_mmr_0", &KDE_Result::mean_mmr_0)
        .def_readwrite("mean_mmr_1", &KDE_Result::mean_mmr_1)
        .def_readwrite("stddev_mmr_0", &KDE_Result::stddev_mmr_0)
        .def_readwrite("stddev_mmr_1", &KDE_Result::stddev_mmr_1);

    py::class_<pd::PreDecodeResult>(m, "PreDecodeResult")
        .def(py::init<>())
        .def_readwrite("decode_result", &pd::PreDecodeResult::decode_result)
        .def_readwrite("nb_rm_edges", &pd::PreDecodeResult::nb_rm_edges);


    //////////// Usergraph Utils bindings ////////////

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
    
    m.def("decode_to_edges_array", &pm::decode_to_edges_array, 
      "Decode a matching graph to an array of edges",
      py::arg("graph"), py::arg("detection_events"));

    m.def("counts_to_det_syndr", &counts_to_det_syndr, 
      "Convert counts to deterministic syndromes",
      py::arg("input_str"), py::arg("_resets") = false, 
      py::arg("verbose") = false, py::arg("reverse") = true);

    m.def("syndromeArrayToDetectionEvents", &syndromeArrayToDetectionEvents, 
      "Convert syndrome array to detection events",
      py::arg("z"), py::arg("num_detectors"), py::arg("boundary_length"));

    m.def("detector_error_model_to_user_graph", &pm::detector_error_model_to_user_graph_private, 
      "Convert a detector error model to a user graph",
      py::arg("detector_error_model"));
    

    //////////// Reweighting bindings ////////////

    m.def("soft_reweight_pymatching", &pm::soft_reweight_pymatching, 
      "Reweight a matching graph using soft information",
      py::arg("matching"), py::arg("not_scaled_IQ_data"), 
      py::arg("synd_rounds"), py::arg("_resets"),
      py::arg("qubit_mapping"), 
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"), 
      py::arg("p_data"), py::arg("p_mixed"), py::arg("common_measure"), 
      py::arg("_adv_probs") = false, py::arg("_bimodal") = false, 
      py::arg("merge_strategy") = "replace", py::arg("p_offset") = 1.0,
      py::arg("p_multiplicator") = 1.0, py::arg("_ntnn_edges") = false);

    m.def("soft_reweight_1Dgauss", &pm::soft_reweight_1Dgauss, 
      "Reweight a matching graph using soft information",
      py::arg("matching"), py::arg("not_scaled_IQ_data"), 
      py::arg("synd_rounds"), py::arg("_resets"),
      py::arg("qubit_mapping"), py::arg("gauss_params_dict"));

    m.def("soft_reweight_kde", &pm::soft_reweight_kde, 
      "Reweight a matching graph using soft information",
      py::arg("matching"), py::arg("not_scaled_IQ_data"), 
      py::arg("synd_rounds"), py::arg("_resets"),
      py::arg("qubit_mapping"), py::arg("kde_dict"));
    
    m.def("reweight_edges_to_one", &pm::reweight_edges_to_one, 
      "Reweight a matching graph to have edge weights of 1",
      py::arg("matching"));

    m.def("reweight_edges_informed", &pm::reweight_edges_informed, 
      "Reweight a matching graph to have edge weights of 1 and use diagonal edges",
      py::arg("matching"), py::arg("distance"), py::arg("p_data"),
      py::arg("p_mixed"), py::arg("p_meas"), py::arg("common_measure") = -1,
      py::arg("_ntnn_edges") = false);
    
    m.def("reweight_edges_based_on_error_probs", &pm::reweight_edges_based_on_error_probs,
      "Reweight a matching graph based on error probabilities",
      py::arg("matching"), py::arg("counts"), py::arg("_resets"), py::arg("method"));

    //////////// Decoding bindings ////////////

    m.def("decode_IQ_shots", &pm::decode_IQ_shots, 
      "Decode a matching graph using IQ data",
      py::arg("matching"), py::arg("not_scaled_IQ_data"), 
      py::arg("synd_rounds"), py::arg("logical"),
      py::arg("_resets"), py::arg("qubit_mapping"), 
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"), 
      py::arg("p_data"), py::arg("p_mixed"), py::arg("common_measure"), 
      py::arg("_adv_probs") = false,
      py::arg("_bimodal") = false, py::arg("merge_strategy") = "replace",
      py::arg("_detailed") = false, py::arg("p_offset") = 1.0, py::arg("p_multiplicator") = 1.0,
      py::arg("_ntnn_edges") = false);

    m.def("decode_IQ_fast", &pm::decode_IQ_fast,
      "Decode a matching graph using IQ data (FAST VERSION)",
      py::arg("detector_error_model"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("logical"),
      py::arg("_resets"), py::arg("qubit_mapping"),
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"),
      py::arg("_detailed") = false,
      py::arg("nb_intervals") = -1,
      py::arg("interval_offset") = 0.5);

    m.def("decode_IQ_1Dgauss", &pm::decode_IQ_1Dgauss,
      "Decode a matching graph using IQ data",
      py::arg("matching"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("logical"),
      py::arg("_resets"), py::arg("qubit_mapping"),
      py::arg("gauss_params_dict"), py::arg("_detailed") = false);

    m.def("decode_IQ_kde", &pm::decode_IQ_kde,
      "Decode a matching graph using IQ data",
      py::arg("detector_error_model"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("logical"),
      py::arg("_resets"), py::arg("qubit_mapping"),
      py::arg("kde_dict"), py::arg("_detailed") = false, 
      py::arg("relError") = -1, py::arg("absError") = -1,
      py::arg("nb_intervals")= -1,
      py::arg("interval_offset") = 0.5);

    m.def("decode_hard_kde", &pm::decode_hard_kde,
    "Hard decoder without reweighting graph or adding edges",
    py::arg("detector_error_model"), py::arg("not_scaled_IQ_data"),
    py::arg("synd_rounds"), py::arg("logical"),
    py::arg("_resets"), py::arg("qubit_mapping"),
    py::arg("kde_dict"), py::arg("_detailed") = false,
    py::arg("relError") = -1, py::arg("absError") = -1,
    py::arg("_ntnn_edges")= false);

    m.def("decode_all_kde", &pm::decode_all_kde,
    py::arg("detector_error_model"), py::arg("not_scaled_IQ_data"),
    py::arg("synd_rounds"), py::arg("logical"),
    py::arg("_resets"), py::arg("qubit_mapping"),
    py::arg("kde_dict"), py::arg("relError") = -1, 
    py::arg("absError") = -1, py::arg("nb_intervals") = -1,
    py::arg("interval_offset") = 0.5);
      
    m.def("decode_IQ_shots_flat", &pm::decode_IQ_shots_flat,
      "Decode a matching graph using IQ data but weight edges to 1",
      py::arg("matching"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("logical"),
      py::arg("_resets"), py::arg("qubit_mapping"),
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"), 
      py::arg("_detailed") = false);
    
    m.def("decode_IQ_shots_flat_informed", &pm::decode_IQ_shots_flat_informed,
      "Decode a matching graph using IQ data but weight edges to 1 and use diagonal edges",
      py::arg("matching"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("logical"),
      py::arg("_resets"), py::arg("qubit_mapping"),
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"),
      py::arg("p_data"), py::arg("p_mixed"), py::arg("p_meas"), py::arg("common_measure") = -1,
      py::arg("_detailed") = false, py::arg("_ntnn_edges") = false);

    m.def("decode_IQ_shots_flat_err_probs", &pm::decode_IQ_shots_flat_err_probs,
      "Reweight and decode a matching graph using error probabilities",
      py::arg("matching"), py::arg("logical"), py::arg("counts_tot"), py::arg("_resets"), py::arg("method"),
      py::arg("not_scaled_IQ_data"), py::arg("synd_rounds"), 
      py::arg("qubit_mapping"), py::arg("kde_grid_dict"), 
      py::arg("scaler_params_dict"), py::arg("_detailed") = false);

    m.def("decode_IQ_shots_no_reweighting", &pm::decode_IQ_shots_no_reweighting,
      "Decode a matching graph using IQ data but do not reweight edges",
      py::arg("matching"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("logical"),
      py::arg("_resets"), py::arg("qubit_mapping"),
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"), py::arg("_detailed") = false);

    //////////// Error probabilities bindings ////////////

    py::class_<ErrorProbabilities>(m, "ErrorProbabilities")
        .def(py::init<>())
        .def_readwrite("probability", &ErrorProbabilities::probability)
        .def_readwrite("samples", &ErrorProbabilities::samples);

    m.def("calculate_naive_error_probs", &calculate_naive_error_probs, 
      "Calculate naive error probabilities",
      py::arg("graph"), py::arg("counts"), py::arg("_resets") = false);

    m.def("calculate_spitz_error_probs", &calculate_spitz_error_probs, 
      "Calculate Spitz error probabilities",
      py::arg("graph"), py::arg("counts"), py::arg("_resets"));


    //////////// KDE bindings ////////////

    m.def("get_KDEs", &get_KDEs, 
      "Get KDEs for each qubit",
      py::arg("all_memories"), py::arg("bandwidths"),
      py::arg("relError") = -1, py::arg("absError") = -1,
      py::arg("num_points") = 51);

    m.def("generate_grid_and_estimate_density", &GenerateGridAndEstimateDensity,
      "Generate grid and estimate density",
      py::arg("kde_dict"), py::arg("num_points"), py::arg("num_std_dev"));    
    
    m.def("get_count_kde", &get_counts_kde, 
      "Get counts using KDE",
      py::arg("not_scaled_IQ_data"), py::arg("qubit_mapping"), 
      py::arg("kde_dict"), py::arg("synd_rounds"));


    //////////// Predecoders bindings ////////////

    m.def("decode_time_nn_predecode_grid", &pd::decode_time_nn_predecode_grid,
      "Decode IQ data using a soft info predecoder",
      py::arg("detector_error_model"), py::arg("not_scaled_IQ_data"),
      py::arg("synd_rounds"), py::arg("logical"),
      py::arg("_resets"), py::arg("qubit_mapping"),
      py::arg("kde_grid_dict"), py::arg("scaler_params_dict"),
      py::arg("_detailed"), 
      py::arg("threshold"),
      py::arg("_ntnn_edges"));

    //////////// Convertor ////////////

    m.def("iqConvertor", &iqConvertor, 
      py::arg("not_scaled_IQ_data"), 
      py::arg("inv_qubit_mapping"), 
      py::arg("kde_dict"), 
      py::arg("relError") = -1.0, 
      py::arg("absError") = -1.0,
      py::arg("handleOutliers") = true);

    m.def("iqConvertorEstim", &iqConvertorEstim, 
      py::arg("not_scaled_IQ_data"), 
      py::arg("inv_qubit_mapping"), 
      py::arg("kde_dict"), 
      py::arg("relError") = -1.0, 
      py::arg("absError") = -1.0);

    m.def("quantizeMatrixVectorized", &quantizeMatrixVectorized, 
      py::arg("matrix"), 
      py::arg("nBits"));

    m.def("quantizeMatrixEntrywise", &quantizeMatrixEntrywise, 
      py::arg("matrix"), 
      py::arg("nBits"));
      
    m.def("decodeConvertorSoft", &decodeConvertorSoft, 
      py::arg("detector_error_model"), 
      py::arg("comparisonMatrix"),
      py::arg("pSoftMatrix"),
      py::arg("synd_rounds"),
      py::arg("logical"),
      py::arg("_resets"),
      py::arg("_detailed"));

    m.def("decodeConvertorAll", &decodeConvertorAll, 
      py::arg("detector_error_model"), 
      py::arg("comparisonMatrix"),
      py::arg("pSoftMatrix"),
      py::arg("synd_rounds"),
      py::arg("logical"),
      py::arg("_resets"),
      py::arg("_detailed")=false,
      py::arg("decode_hard")=false);

    m.def("decodeConvertorDynamicAll", &decodeConvertorDynamicAll,
      py::arg("detector_error_model"),
      py::arg("comparisonMatrix"),
      py::arg("pSoftMatrix"),
      py::arg("msmt_err_dict"),
      py::arg("qubit_mapping"),
      py::arg("synd_rounds"),
      py::arg("logical"),
      py::arg("_resets"),
      py::arg("_detailed") = false,
      py::arg("decode_hard") = false);

    m.def("decodeConvertorAllLeakage", &decodeConvertorAllLeakage,
      py::arg("detector_error_model"),
      py::arg("comparisonMatrix"),
      py::arg("pSoftMatrix"),
      py::arg("synd_rounds"),
      py::arg("logical"),
      py::arg("_resets"),
      py::arg("_detailed") = false,
      py::arg("decode_hard") = false);

    //////////// STIM ////////////

    m.def("decodeStimSoft", &decodeStimSoft, 
      "Decode a STIM circuit",
      py::arg("circuit"),
      py::arg("comparisonMatrix"),
      py::arg("pSoftMatrix"),
      py::arg("synd_rounds"),
      py::arg("logical"),
      py::arg("_resets"),
      py::arg("_detailed") = false);

    m.def("softenCircuit", &softenCircuit, 
      "Modify a STIM circuit",
      py::arg("circuit"),
      py::arg("pSoftRow"));

    m.def("createDetectorErrorModel", &createDetectorErrorModel, 
      "Create a detector error model",
      py::arg("circuit"),
      py::arg("decompose_errors") = false,
      py::arg("flatten_loops") = false,
      py::arg("allow_gauge_detectors") = false,
      py::arg("approximate_disjoint_errors") = false,
      py::arg("ignore_decomposition_failures") = false,
      py::arg("block_decomposition_from_introducing_remnant_edges") = false);

}
