#include "esa_utils.h"

struct RetrievalInputTensor{
    torch::Tensor repre_cache;
    torch::Tensor q_index;
    torch::Tensor repre_index;
    torch::Tensor batch_offset;
    torch::Tensor q_ptrs;
    torch::Tensor workspace;
    int num_q_heads;
    int batch_size;
};

struct RetrievalOutputTensor{
    torch::Tensor score;
    torch::Tensor score_sorted;
    torch::Tensor index_ranged;
    torch::Tensor index_sorted;
};

void esa_repre(torch::Tensor key_cache, torch::Tensor repre_cache, torch::Tensor block_table, torch::Tensor repre_table);

void esa_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, torch::Tensor workspace);

void esa_retrieval(RetrievalInputTensor input, RetrievalOutputTensor output);
