#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>

// kernel to initialize indices [0..N)
__global__ void init_indices(int* idx, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) idx[i] = i;
}

int main() {
    const int N = 1 << 20;
    const int K = 10;
    // 1) generate random data
    std::vector<float> h_data(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_data[i] = dist(rng);

    // 2) allocate and copy to device
    float* d_data;
    int*   d_indices;
    float* d_sorted_vals;
    int*   d_sorted_idx;
    cudaMalloc(&d_data,       N * sizeof(float));
    cudaMalloc(&d_indices,    N * sizeof(int));
    cudaMalloc(&d_sorted_vals,N * sizeof(float));
    cudaMalloc(&d_sorted_idx, N * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 3) init device indices
    const int TPB = 256;
    int blocks = (N + TPB - 1) / TPB;
    init_indices<<<blocks, TPB>>>(d_indices, N);
    cudaDeviceSynchronize();

    // 4) run CUB segmented radix sort (one segment)
    int h_offsets[2] = { 0, N };
    int* d_offsets;
    cudaMalloc(&d_offsets, 2 * sizeof(int));
    cudaMemcpy(d_offsets, h_offsets, 2 * sizeof(int), cudaMemcpyHostToDevice);
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_data,  d_sorted_vals,
        d_indices, d_sorted_idx,
        N, 1, d_offsets, d_offsets + 1);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_data,  d_sorted_vals,
        d_indices, d_sorted_idx,
        N, 1, d_offsets, d_offsets + 1);

    // 5) copy top-K indices back
    std::vector<int> h_topk(K);
    cudaMemcpy(h_topk.data(), d_sorted_idx, K * sizeof(int), cudaMemcpyDeviceToHost);

    // 6) compute CPU ground-truth
    std::vector<int> gt(N);
    for (int i = 0; i < N; ++i) gt[i] = i;
    std::partial_sort(gt.begin(), gt.begin() + K, gt.end(),
                      [&h_data](int a, int b) { return h_data[a] > h_data[b]; });

    // 7) compare
    bool ok = true;
    for (int i = 0; i < K; ++i) {
        if (h_topk[i] != gt[i]) { ok = false; break; }
    }
    std::cout << (ok ? "PASS\n" : "FAIL\n");
    
    // 8) cleanup
    cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_sorted_vals);
    cudaFree(d_sorted_idx);
    cudaFree(d_offsets);
    cudaFree(d_temp);

    return ok ? 0 : 1;
}
