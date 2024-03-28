#ifndef STIMUTILS_H
#define STIMUTILS_H

#include <Eigen/Dense>  
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"

#include "../PyMatching/matching_graph.h"
#include "../PyMatching/user_graph_utils.h"


///// StimDecoder /////

pm::DetailedDecodeResult decodeStimSoft(
    stim::Circuit circuit,
    Eigen::MatrixXi comparisonMatrix,
    Eigen::MatrixXd pSoftMatrix,
    int synd_rounds,
    int logical,
    bool _resets,
    bool _detailed = false);


///// StimUtils /////

bool needs_modification(const stim::CircuitInstruction& instruction);

stim::SpanRef<double> allocate_args_in_buffer(stim::MonotonicBuffer<double>& arg_buf, const std::vector<double>& new_args);

stim::SpanRef<const stim::GateTarget> allocate_targets_in_buffer(stim::MonotonicBuffer<stim::GateTarget>& target_buf, 
                                                                 const std::vector<stim::GateTarget>& new_targets);

void softenCircuit(stim::Circuit& circuit, const Eigen::VectorXd& pSoftRow);

stim::DetectorErrorModel createDetectorErrorModel(const stim::Circuit& circuit,
                                        bool decompose_errors = false,
                                        bool flatten_loops = false,
                                        bool allow_gauge_detectors = false,
                                        double approximate_disjoint_errors = false,
                                        bool ignore_decomposition_failures = false,
                                        bool block_decomposition_from_introducing_remnant_edges = false);





#endif // STIMUTILS_H


