#include <omp.h>
#include <iostream>

int main() {
    #pragma omp parallel
    std::cout << "Thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << std::endl;
    return 0;
}
