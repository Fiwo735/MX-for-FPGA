#ifndef LINEAR_CUH
#define LINEAR_CUH

#include <iostream>
#include <torch/extension.h>
#include "round_fp.cuh"

#define TILE_SIZE 16
#define NUM_BINS 16
#define TILE_SIZE_2 32



template <typename scalar_t>
__global__ void ordmm_chunk_bcast_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int in_batch, int in_features, int out_features,
    int man_width, int exp_width
){

    int col = blockIdx.x * TILE_SIZE_2 + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE_2 + threadIdx.y;
    int prt = blockIdx.z;

    // Shared memory for tiles of input and weight
    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];

    float acc = 0;

    // Loop over the tiles of the input in steps of TILE_SIZE_2
    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
        // Collaborative loading of tiles into shared memory
        const int input_col = t * TILE_SIZE_2 + threadIdx.x;
        const int weight_row = t * TILE_SIZE_2 + threadIdx.y;

        if (row < in_batch && input_col < in_features){
            shared_A[threadIdx.y][threadIdx.x] = input[prt * in_batch * in_features + row * in_features + input_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }
        if (col < out_features && weight_row < in_features){
            shared_B[threadIdx.y][threadIdx.x] = weight[prt * out_features * in_features + col * in_features + weight_row];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        float local_acc = 0;

        // Perform the multiplication for this tile
        for (int k=0; k < TILE_SIZE_2; ++k){
            local_acc += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        acc += local_acc;
        acc = round_rne_fp_full(acc, man_width, exp_width);

        __syncthreads();
    }
    if (row < in_batch && col < out_features){
        output[prt * in_batch * out_features + row * out_features + col] = acc;
    }
}

template <typename scalar_t>
__global__ void ordmm_chunk_full_quant_bcast_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int in_batch, int in_features, int out_features,
    int man_width, int exp_width
){

    int col = blockIdx.x * TILE_SIZE_2 + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE_2 + threadIdx.y;
    int prt = blockIdx.z;

    // Shared memory for tiles of input and weight
    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_C[TILE_SIZE_2][TILE_SIZE_2];

    float acc;

    // Loop over the tiles of the input in steps of TILE_SIZE_2
    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
        // Collaborative loading of tiles into shared memory
        const int input_col = t * TILE_SIZE_2 + threadIdx.x;
        const int weight_row = t * TILE_SIZE_2 + threadIdx.y;

        if (row < in_batch && input_col < in_features){
            shared_A[threadIdx.y][threadIdx.x] = input[prt * in_batch * in_features + row * in_features + input_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }
        if (col < out_features && weight_row < in_features){
            shared_B[threadIdx.y][threadIdx.x] = weight[prt * out_features * in_features + col * in_features + weight_row];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }
        if (row < in_batch && col < out_features){
            shared_C[threadIdx.y][threadIdx.x] = output[prt * in_batch * out_features + row * out_features + col];
        } else {
            shared_C[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        acc = 0;

        // Perform the multiplication for this tile
        for (int k=0; k < TILE_SIZE_2; ++k){
            acc += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            acc = round_rne_fp_full(acc, man_width, exp_width);
        }

        acc += shared_C[threadIdx.y][threadIdx.x];
        acc = round_rne_fp_full(acc, man_width, exp_width);
        if (row < in_batch && col < out_features){
            output[prt * in_batch * out_features + row * out_features + col] = acc;
        }

        __syncthreads();
    }
}

torch::Tensor ordmm_chunk_bcast(
    torch::Tensor input,
    torch::Tensor weight_tpose,
    int man_width, int exp_width,
    bool full_quant=false
){
    // Broadcast tensors to compatible batch shapes
    auto batch_shape = torch::infer_size(
        input.sizes().slice(0, input.dim() - 2),
        weight_tpose.sizes().slice(0, weight_tpose.dim() - 2)
    );
    std::vector<int64_t> input_expanded_shape = batch_shape; // Start with the batch shape
    auto input_last_dims = input.sizes().slice(input.dim() - 2, 2); // Get the last two dimensions
    input_expanded_shape.insert(input_expanded_shape.end(), input_last_dims.begin(), input_last_dims.end()); // Concatenate the last dimensions
    input = input.expand(input_expanded_shape); // Expand to the new shape
    std::vector<int64_t> weight_tpose_expanded_shape = batch_shape; // Start with the batch shape
    auto weight_tpose_last_dims = weight_tpose.sizes().slice(weight_tpose.dim() - 2, 2); // Get the last two dimensions
    weight_tpose_expanded_shape.insert(weight_tpose_expanded_shape.end(), weight_tpose_last_dims.begin(), weight_tpose_last_dims.end()); // Concatenate the last dimensions
    weight_tpose = weight_tpose.expand(weight_tpose_expanded_shape); // Expand to the new shape
    // Flatten batch dimensions for looping
    int64_t batch_size = std::accumulate(batch_shape.begin(), batch_shape.end(), 1L, std::multiplies<int64_t>());
    auto input_flat = input.reshape({batch_size, input.size(-2), input.size(-1)}).to(weight_tpose.dtype());;
    auto weight_tpose_flat = weight_tpose.reshape({batch_size, weight_tpose.size(-2), weight_tpose.size(-1)});

    std::vector<int64_t> target_shape = input.sizes().slice(0, input.sizes().size() - 1).vec();
    target_shape.push_back(weight_tpose.size(-2));

    input_flat = input_flat.contiguous();
    weight_tpose_flat = weight_tpose_flat.contiguous();

    int part = input_flat.size(0);
    int in_batch = input_flat.size(1);
    int in_features = input_flat.size(2);
    int out_features = weight_tpose_flat.size(1);

    torch::Tensor output = torch::zeros({part, in_batch, out_features}, input.options());

    dim3 block_dim(TILE_SIZE_2, TILE_SIZE_2);
    dim3 grid_dim((out_features + TILE_SIZE_2 - 1) / TILE_SIZE_2, (in_batch + TILE_SIZE_2 - 1) / TILE_SIZE_2, part);

    if(!full_quant){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "matmul_chunk", ([&]{
            ordmm_chunk_bcast_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                weight_tpose_flat.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                in_batch,
                in_features,
                out_features,
                man_width,
                exp_width
            );
        }));
    }else{
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "matmul_chunk", ([&]{
            ordmm_chunk_full_quant_bcast_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                weight_tpose_flat.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                in_batch,
                in_features,
                out_features,
                man_width,
                exp_width
            );
        }));
    }
    cudaDeviceSynchronize();

    return output.view(target_shape).to(input.dtype());
}


#endif // LINEAR_CUH
