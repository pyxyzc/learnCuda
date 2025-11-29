import cupy as cp
import torch

# compile and load the CUDA kernel
module = cp.RawModule(path='repre_compute.cu', backend='nvcc', options=('--std=c++14',))
extract_repre_kernel = module.get_function('extract_repre')

def extract_repre_gpu(key_cache, repre_cache, block_table, block_table_2):
    """
    GPU wrapper for extract_repre kernel.

    key_cache: cupy.ndarray, shape (N, block_size, dim), dtype float32
    block_table, block_table_2: cupy.ndarray, shape (block_number,), dtype int32

    Returns:
      repre_cache: cupy.ndarray, shape (N, dim), dtype float32
    """
    # ensure correct types
    key_cache = cp.asarray(key_cache, dtype=cp.float32)
    repre_cache = cp.asarray(repre_cache, dtype=cp.float32)
    block_table = cp.asarray(block_table, dtype=cp.int32)
    block_table_2 = cp.asarray(block_table_2, dtype=cp.int32)

    N, block_size, dim = key_cache.shape
    block_number = block_table.size

    # launch kernel: grid = block_number, block = dim
    extract_repre_kernel(
        (block_number,), (dim,),
        (key_cache, repre_cache, block_table, block_table_2, block_size, dim)
    )
    return repre_cache


key_cache = torch.randn(1000, 128, 576, dtype = torch.float32).cuda()

repre_cache = torch.randn(1000, 576, dtype = torch.float32).cuda()

block_table = torch.arange(0, 40, dtype = torch.int32).cuda()

block_table_2 = torch.arange(0, 40, dtype = torch.int32).cuda()

extract_repre_gpu(key_cache, repre_cache, block_table, block_table_2)
