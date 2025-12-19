#pragma once
#include "esa_utils.h"

#define CUDA_TRANS_UNIT_SIZE (sizeof(uint4) * 2)
#define CUDA_TRANS_BLOCK_NUMBER (32)
#define CUDA_TRANS_BLOCK_SIZE (256)
#define CUDA_TRANS_THREAD_NUMBER (CUDA_TRANS_BLOCK_NUMBER * CUDA_TRANS_BLOCK_SIZE)

void esa_copy(torch::Tensor src, torch::Tensor dst, size_t size);
