#include "esa_utils.h"

/**
 * This kernel performs: repre_cache[repre_block_table[i]] = mean( key_cache[key_block_table[i]], 0 )
 *
 * @param key_cache: [N, block_size, dim]
 * @param repre_cache: [N, dim]
 * @param key_block_table: [S]
 * @param repre_block_table: [S]
 */
template <typename scalar_t>
__global__ void extract_repre(const scalar_t *key_cache, scalar_t *repre_cache, const int *key_block_table, const int *repre_block_table, int block_size, int dim) {
    int idx = blockIdx.x;
    int block_id = key_block_table[idx];
    int block_id_2 = repre_block_table[idx];
    const scalar_t* key_ptr = key_cache + block_id * block_size * dim;
    scalar_t* repre_ptr = repre_cache + block_id_2 * dim;
    int d = threadIdx.x;
    if (d < dim) {
        float sum = 0;
        for (int j = 0; j < block_size; ++j) {
            sum += static_cast<float>(key_ptr[j * dim + d]);
        }
        repre_ptr[d] = static_cast<scalar_t>(sum / block_size);
    }
}

/**
 * This kernel performs: score[i] = queries[batch_index[i]] * repre_cache[block_table[i]]
 *
 * @param queries: a list of tensors. { batch_size * [num_q_heads, dim] }
 * @param repre_cache: [N, num_k_heads, dim]
 * @param score: [S]
 * @param block_table: [S]
 * @param batch_index: [S]
 */

__global__ void retrieval_kernel_fp16(__half **queries, __half *__restrict__ repre_cache, __half *__restrict__ score, int *__restrict__ block_table, int *__restrict__ batch_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries[batch_index[blockIdx.x]];
    auto *k_offset = repre_cache + block_table[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += __half2float(q_val) * __half2float(k_val);
            }
        }
    }
    // printf("block_id: %d, threadIdx.y: %d, threadIdx.x: %d, sum: %f\n", block_table[blockIdx.x], threadIdx.y, threadIdx.x, sum);
    // score[blockIdx.x * 32 * 32 + threadIdx.y * 32 + threadIdx.x] = __float2half(sum);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = __float2half(sum);
        }
    }
}


__global__ void retrieval_kernel_fp32(float **queries, float *__restrict__ repre_cache, float *__restrict__ score, int *__restrict__ block_table, int *__restrict__ batch_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries[batch_index[blockIdx.x]];
    auto *k_offset = repre_cache + block_table[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += q_val * k_val;
            }
        }
    }
    // printf("block_id: %d, threadIdx.x: %d, threadIdx.y: %d, sum: %f\n", block_table[blockIdx.x], threadIdx.x, threadIdx.y, sum);
    // score[blockIdx.x * 32 * 32 + threadIdx.y * 32 + threadIdx.x] = sum;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = sum;
        }
    }
}

__global__ void retrieval_kernel_bf16(__nv_bfloat16 **queries, __nv_bfloat16 *__restrict__ repre_cache, __nv_bfloat16 *__restrict__ score, int *__restrict__ block_table, int *__restrict__ batch_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries[batch_index[blockIdx.x]];
    auto *k_offset = repre_cache + block_table[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += __bfloat162float(q_val) * __bfloat162float(k_val);
            }
        }
    }
    // printf("block_id: %d, threadIdx.x: %d, threadIdx.y: %d, sum: %f\n", block_table[blockIdx.x], threadIdx.x, threadIdx.y, sum);
    // score[blockIdx.x * 32 * 32 + threadIdx.y * 32 + threadIdx.x] = sum;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = __float2bfloat16(sum);
        }
    }
}
