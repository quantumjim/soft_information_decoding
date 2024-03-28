#include "StimUtils.h"


// pm::DetailedDecodeResult decodeStimSoft(
//     stim::Circuit circuit,
//     Eigen::MatrixXi comparisonMatrix,
//     Eigen::MatrixXd pSoftMatrix,
//     int synd_rounds,
//     int logical,
//     bool _resets,
//     bool _detailed) {
        
//     pm::DetailedDecodeResult result;
//     result.num_errors = 0;

//     std::set<size_t> empty_set;        
//     int distance = (comparisonMatrix.cols() + synd_rounds) / (synd_rounds + 1); // Adapted for RepCodes

//     #pragma omp parallel 
//     {
//         stim::Circuit circuitCopy = circuit;
//         // Eventually copy before each softening to make sure no error happen!

//         #pragma omp for nowait  
//         for (int shot = 0; shot < comparisonMatrix.rows(); ++shot) {
//             std::string count_key;

//             // Soften the circuit
//             softenCircuit(circuitCopy, pSoftMatrix.row(shot));
//             stim::DetectorErrorModel errorModel = createDetectorErrorModel(circuitCopy);
//             pm::UserGraph matching = pm::detector_error_model_to_user_graph_private(errorModel);

//             for (int msmt = 0; msmt < comparisonMatrix.cols(); ++msmt) {

//                 // Counts
//                 int measurementResult = comparisonMatrix(shot, msmt);
//                 if (measurementResult == 1) {
//                     count_key += "1";
//                 } else {
//                     count_key += "0";
//                 }
//                 if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
//                     count_key += " ";
//                 }  
//             }

//             // Decoding
//             auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false); // Not optimal but good enough for now
//             auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
//             auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
//             int actual_observable = (static_cast<int>(count_key.back()) - logical) % 2;

//             // Result saving
//             if (_detailed) {
//                 pm::ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
//                 result.error_details[shot] = errorDetail;
//             }
//             #pragma omp critical
//             {
//                 if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
//                     result.num_errors++; // Increment error count if they don't match
//                     result.indices.push_back(shot);
//                 }
//             }
//         }
//     }
//     return result;
// }

pm::DetailedDecodeResult decodeStimSoft(
    stim::Circuit circuit,
    Eigen::MatrixXi comparisonMatrix,
    Eigen::MatrixXd pSoftMatrix,
    int synd_rounds,
    int logical,
    bool _resets,
    bool _detailed) {
    
    auto start_total = std::chrono::high_resolution_clock::now();

    pm::DetailedDecodeResult result;
    result.num_errors = 0;

    std::set<size_t> empty_set;        
    int distance = (comparisonMatrix.cols() + synd_rounds) / (synd_rounds + 1); // Adapted for RepCodes

    #pragma omp parallel 
    {
        stim::Circuit circuitCopy;

        #pragma omp for nowait  
        for (int shot = 0; shot < comparisonMatrix.rows(); ++shot) {
            auto start_copy = std::chrono::high_resolution_clock::now();
            circuitCopy = circuit;
            auto end_copy = std::chrono::high_resolution_clock::now();

            auto start_soften = std::chrono::high_resolution_clock::now();
            Eigen::VectorXd pSoftRowCopy = pSoftMatrix.row(shot); // Copy to address non-const lvalue reference issue
            softenCircuit(circuitCopy, pSoftRowCopy);
            auto end_soften = std::chrono::high_resolution_clock::now();

            auto start_error_model = std::chrono::high_resolution_clock::now();
            stim::DetectorErrorModel errorModel = createDetectorErrorModel(circuitCopy);
            auto end_error_model = std::chrono::high_resolution_clock::now();

            auto start_matching = std::chrono::high_resolution_clock::now();
            pm::UserGraph matching = pm::detector_error_model_to_user_graph_private(errorModel);
            auto end_matching = std::chrono::high_resolution_clock::now();

            std::string count_key;
            for (int msmt = 0; msmt < comparisonMatrix.cols(); ++msmt) {
                // Processing counts
                int measurementResult = comparisonMatrix(shot, msmt);
                if (measurementResult == 1) {
                    count_key += "1";
                } else {
                    count_key += "0";
                }
                if ((msmt + 1) % (distance - 1) == 0 && (msmt + 1) / (distance - 1) <= synd_rounds) {
                    count_key += " ";
                }
            }

            auto start_det_syndromes = std::chrono::high_resolution_clock::now();
            auto det_syndromes = counts_to_det_syndr(count_key, _resets, false, false);
            auto end_det_syndromes = std::chrono::high_resolution_clock::now();

            auto start_detection_events = std::chrono::high_resolution_clock::now();
            auto detectionEvents = syndromeArrayToDetectionEvents(det_syndromes, matching.get_num_detectors(), matching.get_boundary().size());
            auto end_detection_events = std::chrono::high_resolution_clock::now();

            auto start_decoding = std::chrono::high_resolution_clock::now();
            auto [predicted_observables, rescaled_weight] = decode(matching, detectionEvents);
            auto end_decoding = std::chrono::high_resolution_clock::now();

            int actual_observable = (static_cast<int>(count_key.back()) - logical) % 2;

            auto start_saving = std::chrono::high_resolution_clock::now();
            if (_detailed) {
                pm::ShotErrorDetails errorDetail = createShotErrorDetails(matching, detectionEvents, det_syndromes);
                result.error_details[shot] = errorDetail;
            }
            #pragma omp critical
            {
                if (!predicted_observables.empty() && predicted_observables[0] != actual_observable) {
                    result.num_errors++; // Increment error count if they don't match
                    result.indices.push_back(shot);
                }
            }
            auto end_saving = std::chrono::high_resolution_clock::now();

            #pragma omp critical // Ensure one thread at a time outputs to avoid jumbled logs
            {
                std::cout << "Timings for shot " << shot << " (seconds):" << std::endl;
                std::cout << "\tCopying: " << std::chrono::duration<double>(end_copy - start_copy).count() << std::endl;
                std::cout << "\tSoftening: " << std::chrono::duration<double>(end_soften - start_soften).count() << std::endl;
                std::cout << "\tError Model Creation: " << std::chrono::duration<double>(end_error_model - start_error_model).count() << std::endl;
                std::cout << "\tMatching Graph Creation: " << std::chrono::duration<double>(end_matching - start_matching).count() << std::endl;
                std::cout << "\tDet Syndromes: " << std::chrono::duration<double>(end_det_syndromes - start_det_syndromes).count() << std::endl;
                std::cout << "\tDetection Events: " << std::chrono::duration<double>(end_detection_events - start_detection_events).count() << std::endl;
                std::cout << "\tDecoding: " << std::chrono::duration<double>(end_decoding - start_decoding).count() << std::endl;
                std::cout << "\tSaving: " << std::chrono::duration<double>(end_saving - start_saving).count() << std::endl;
            }
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::cout << "Total execution time (seconds): " << std::chrono::duration<double>(end_total - start_total).count() << std::endl;

    return result;
}

