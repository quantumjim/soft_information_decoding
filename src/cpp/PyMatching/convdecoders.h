#ifndef CONVDECODERS_H
#define CONVDECODERS_H

#include "user_graph_utils.h"
#include "matching_graph.h"
#include "../Probabilities/probabilities.h"


pm::DetailedDecodeResult decodeConvertorSoft(
    stim::DetectorErrorModel& detector_error_model,
    Eigen::MatrixXi comparisonMatrix,
    Eigen::MatrixXd pSoftMatrix,
    int synd_rounds,
    int logical,
    bool _resets,
    bool _detailed = false);

pm::DetailedDecodeResult decodeConvertorAll(
    stim::DetectorErrorModel& detector_error_model,
    Eigen::MatrixXi comparisonMatrix,
    Eigen::MatrixXd pSoftMatrix,
    int synd_rounds,
    int logical,
    bool _resets,
    bool _detailed = false,
    bool decode_hard = false); 



#endif // CONVDECODERS_H