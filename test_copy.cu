//  1 一次大拷贝能跑满 PCIe 的峰值带宽。
//  2 多次小拷贝带宽急剧下降，并且 (t_small – t_big)/N 给出了每个 cudaMemcpy 的 CPU 启动/调度开销。
// 这样就能验证：大量小块 cudaMemcpy 的调用开销（CPU 侧 latency）在拖累总体带宽，而不是 PCIe 本身的传输速率问题。
// 结果：55GB/s vs 38GB/s
// 单个cudaMemcpy指令的开销大约9us


#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
      std::exit(1);                                                  \
    }                                                                 \
  } while (0)

int main() {
  using clock = std::chrono::high_resolution_clock;
  size_t total_bytes = 1ULL << 30;  // 1 GB
  int N;
  scanf("%d", &N);
  size_t chunk_bytes = total_bytes / N;
  // chunk_bytes *= 4;
  // chunk_bytes /= (256);
  printf("total copy number: %d\n", N);

  // 分配页锁定主机内存和设备内存
  void* hptr = nullptr;
  void* dptr = nullptr;
  CUDA_CHECK(cudaMallocHost(&hptr, total_bytes));
  CUDA_CHECK(cudaMalloc(&dptr, total_bytes));

  // 预热
  CUDA_CHECK(cudaMemcpy(dptr, hptr, total_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  // —— Test 1: 一次大拷贝 ——
  auto t0 = clock::now();
  CUDA_CHECK(cudaMemcpyAsync(dptr, hptr, total_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = clock::now();
  double t_big = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "ONE big copy: "
            << (double)total_bytes / t_big / 1e9
            << " GB/s\n";

  // —— Test 2: N 次小拷贝 ——
  t0 = clock::now();


  double arr[N];
  for (int i = 0; i < N; i++) {
    auto x = clock::now();
    CUDA_CHECK(cudaMemcpyAsync(
      (char*)dptr + i*chunk_bytes,
      (char*)hptr + i*chunk_bytes,
      chunk_bytes,
      cudaMemcpyHostToDevice));
    auto y = clock::now();
    double duration = std::chrono::duration<double>(y - x).count();
    arr[i] = duration;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  t1 = clock::now();
  double t_small = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "small: " << t_small << 
    " big: " << t_big << std::endl;
  std::cout << N << " small copies: "
            << (double)total_bytes / t_small / 1e9
            << " GB/s\n";

  // 估算每次 memcpy 的启动开销（us）
  double overhead_us = (t_small - t_big) / N * 1e6;
  double overhead_us_2 = 0.0f;
  for(int i = 0; i < N; ++i) {
    overhead_us_2 += arr[i];
  }
  overhead_us_2 /= N;
  overhead_us_2 *= 1e6;
  std::cout << "Approx overhead per cudaMemcpy call: "
            << overhead_us << " us, " << overhead_us_2 <<" us\n";

  // 清理
  CUDA_CHECK(cudaFree(dptr));
  CUDA_CHECK(cudaFreeHost(hptr));
  return 0;
}


