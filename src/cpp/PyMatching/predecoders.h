#ifndef PREDECODERS_H
#define PREDECODERS_H

#include "user_graph_utils.h"
#include "matching_graph.h"
#include "../Probabilities/probabilities.h"


namespace pd {
    pm::DetailedDecodeResult decode_time_nn_predecode_grid(
            stim::DetectorErrorModel detector_error_model,
            const Eigen::MatrixXcd &not_scaled_IQ_data,
            int synd_rounds,
            int logical,
            bool _resets,
            const std::map<int, int> &qubit_mapping,
            const std::map<int, GridData> &kde_grid_dict,
            const std::map<int, std::pair<std::pair<double, double>, std::pair<double, double>>> &scaler_params_dict,
            bool _detailed,
            float threshold=1.0); // defaults to 1.0 so that it is always below
}



#endif // PREDECODERS_H