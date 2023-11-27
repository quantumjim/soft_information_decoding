#include "matching_graph.h"
#include "user_graph_utils.h"
#include <iostream>


// void print_edges_of_graph(const pm::UserGraph& graph) {
//     auto edges = get_edges(graph);

//     for (const auto& edge : edges) {
//         std::cout << "Edge between node " << edge.node1;
//         if (edge.node2 == SIZE_MAX) {
//             std::cout << " and boundary";
//         } else {
//             std::cout << " and node " << edge.node2;
//         }
//         std::cout << " - Weight: " << edge.attributes.weight
//                   << ", Error Probability: " << edge.attributes.error_probability
//                   << ", Fault IDs: {";

//         // Print fault IDs
//         for (auto it = edge.attributes.fault_ids.begin(); it != edge.attributes.fault_ids.end(); ++it) {
//             if (it != edge.attributes.fault_ids.begin()) std::cout << ", ";
//             std::cout << *it;
//         }

//         std::cout << "}" << std::endl;
//     }
// }

void processGraph_test(pm::UserGraph& graph) {
    std::cout << "Processing graph" << std::endl;
    // Implement the logic for processing the graph
}


// Comand to compile:
// cd src/cpp/Pymatching
// g++ matching_graph.cpp -o matching_graph \
// -I"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/src" \
// -I"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/Stim/src" \
// -L"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/bazel-bin" \
// -llibpymatching -std=c++20
