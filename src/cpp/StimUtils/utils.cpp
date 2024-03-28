#include "StimUtils.h"


bool needs_modification(const stim::CircuitInstruction& instruction) {
    return instruction.gate_type == stim::GateType::M || instruction.gate_type == stim::GateType::MR;
}

stim::SpanRef<double> allocate_args_in_buffer(stim::MonotonicBuffer<double>& arg_buf, const std::vector<double>& new_args) {
    // Append each argument value to the buffer's tail and then commit.
    for (const double& arg : new_args) {
        arg_buf.append_tail(arg);
    }
    // Commit the tail to the buffer and return a SpanRef to this committed segment.
    return arg_buf.commit_tail();
}

stim::SpanRef<const stim::GateTarget> allocate_targets_in_buffer(stim::MonotonicBuffer<stim::GateTarget>& target_buf, const std::vector<stim::GateTarget>& new_targets) {
    // Append each target to the buffer's tail
    for (const auto& target : new_targets) {
        target_buf.append_tail(target);
    }
    // Commit the tail to the buffer and return a SpanRef to this committed segment
    return target_buf.commit_tail();
}

void softenCircuit(stim::Circuit& circuit, const Eigen::VectorXd& pSoftRow) {
    circuit = circuit.flattened(); // Flatten the circuit because pSofts vary in the repeat blocks
    int msmtIdx = -1;
    for (size_t i = 0; i < circuit.operations.size(); ++i) {
        stim::CircuitInstruction& instruction = circuit.operations[i];
        if (needs_modification(instruction)) {
            std::vector<stim::CircuitInstruction> new_instructions;
            for (auto target : instruction.targets) {
                msmtIdx++;
                double pSoft = pSoftRow[msmtIdx];
                std::vector<double> new_args = {pSoft};
                stim::SpanRef<double> new_args_ref = allocate_args_in_buffer(circuit.arg_buf, new_args);
                stim::SpanRef<const stim::GateTarget> target_ref = allocate_targets_in_buffer(circuit.target_buf, {target});
                stim::CircuitInstruction new_instruction(instruction.gate_type, new_args_ref, target_ref);
                new_instructions.push_back(new_instruction);
            }
            // Remove the MR instruction after adding new instructions
            circuit.operations.erase(circuit.operations.begin() + i);
            // Insert the new instructions at the current position
            circuit.operations.insert(circuit.operations.begin() + i, new_instructions.begin(), new_instructions.end());
            // Adjust the index to skip the newly added instructions
            i += new_instructions.size() - 1;
        }
    }
}


stim::DetectorErrorModel createDetectorErrorModel(const stim::Circuit& circuit,
                                        bool decompose_errors,
                                        bool flatten_loops,
                                        bool allow_gauge_detectors,
                                        double approximate_disjoint_errors,
                                        bool ignore_decomposition_failures,
                                        bool block_decomposition_from_introducing_remnant_edges) {
// Call the ErrorAnalyzer::circuit_to_detector_error_model function
return stim::ErrorAnalyzer::circuit_to_detector_error_model(
            circuit,
            decompose_errors,
            !flatten_loops,
            allow_gauge_detectors,
            approximate_disjoint_errors,
            ignore_decomposition_failures,
            block_decomposition_from_introducing_remnant_edges);
}