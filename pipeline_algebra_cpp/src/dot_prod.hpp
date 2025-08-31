#ifndef DOT_PROD_HPP
#define DOT_PROD_HPP

#include "pipeline_utils.hpp" // Or any other necessary headers
#include <chrono>

// You can't declare a __global__ kernel, but you can declare the wrapper
// function
Matrix compute_kernel_block_batched_gpu(
    int c, int i, int s, const std::vector<FirstHopState> &firstHopStates,
    const std::vector<int> &node_ids, const GraphData &data,
    const std::vector<int> &nodeDegrees, const std::vector<int> &nodeDegreesC1,
    int nPower = 1, int batch_size = -1);

#endif // DOT_PROD_HPP
