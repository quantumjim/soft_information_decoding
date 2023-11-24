#include "pymatching/rand/rand_gen.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"

int main() {
    try {
        // Assuming UserGraph has a default constructor.
        // If it requires parameters, provide them as necessary.
        pm::UserGraph userGraph;

        // If MatchingGraph is a member of UserGraph, access it directly or via a getter method.
        // This part is dependent on how UserGraph exposes MatchingGraph.
        // Example: pm::MatchingGraph& matchingGraph = userGraph.getMatchingGraph();

        // If UserGraph does not expose MatchingGraph directly, you might need to
        // perform additional steps as per the library's design.

        // Use matchingGraph as needed...
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


// Comand to compile:
// g++ matching_graph.cpp -o matching_graph \
// -I"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/src" \
// -I"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/Stim/src" \
// -L"/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/Soft-Info/libs/PyMatching/bazel-bin" \
// -llibpymatching -std=c++20
