#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)                   \
                << " (" #call ") at " << __FILE__ << ":" << __LINE__           \
                << std::endl;                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main() {
  using clock = std::chrono::high_resolution_clock;
  size_t total_bytes = 1ULL << 30; // 1 GiB
  size_t chunk_bytes = 1ULL << 20; // 1 MiB
  chunk_bytes /= 8;
  const int N = total_bytes / chunk_bytes;
  const int M = 16; // try 2 or 4 streams

  // allocate pinned host + device buffers
  void *h, *d;
  CUDA_CHECK(cudaMallocHost(&h, total_bytes));
  CUDA_CHECK(cudaMalloc(&d, total_bytes));

  // warm-up
  CUDA_CHECK(cudaMemcpy(d, h, total_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  // 1) single‐stream async
  cudaStream_t s1;
  CUDA_CHECK(cudaStreamCreate(&s1));
  auto t0 = clock::now();
  for (int i = 0; i < N; i++) {
    CUDA_CHECK(cudaMemcpyAsync((char *)d + i * chunk_bytes,
                               (char *)h + i * chunk_bytes, chunk_bytes,
                               cudaMemcpyHostToDevice, s1));
  }
  CUDA_CHECK(cudaStreamSynchronize(s1));
  auto t1 = clock::now();
  double t_single = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "Single‐stream async bandwidth: "
            << (double)total_bytes / t_single / 1e9 << " GB/s\n";
  CUDA_CHECK(cudaStreamDestroy(s1));

  // 2) multi‐stream async
  std::vector<cudaStream_t> streams(M);
  for (int i = 0; i < M; i++)
    CUDA_CHECK(cudaStreamCreate(&streams[i]));

  t0 = clock::now();
  for (int i = 0; i < N; i++) {
    int s = i % M;
    CUDA_CHECK(cudaMemcpyAsync((char *)d + i * chunk_bytes,
                               (char *)h + i * chunk_bytes, chunk_bytes,
                               cudaMemcpyHostToDevice, streams[s]));
  }
  for (auto &st : streams)
    CUDA_CHECK(cudaStreamSynchronize(st));
  t1 = clock::now();
  double t_multi = std::chrono::duration<double>(t1 - t0).count();
  std::cout << M << "‐stream async bandwidth: "
            << (double)total_bytes / t_multi / 1e9 << " GB/s\n";

  // cleanup
  for (auto &st : streams)
    CUDA_CHECK(cudaStreamDestroy(st));
  CUDA_CHECK(cudaFree(d));
  CUDA_CHECK(cudaFreeHost(h));
  return 0;
}
