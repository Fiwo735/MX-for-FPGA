#ifndef LINEAR_CUH_SCALED
#define LINEAR_CUH_SCALED

#include <iostream>
#include <torch/extension.h>
#include "round_fp.cuh"

#define TILE_SIZE 16
#define NUM_BINS 16
#define TILE_SIZE_2 32



template <typename scalar_t>
__global__ void ordmm_chunk_comp_sum_bcast_scaled_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float* __restrict__ scale_input,
    const float* __restrict__ scale_weight,
    float* __restrict__ output,
    int in_batch, int in_features, int out_features,
    int man_width, int exp_width
){

    int col = blockIdx.x * TILE_SIZE_2 + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE_2 + threadIdx.y;
    int prt = blockIdx.z;

    // Shared memory for tiles of input and weight
    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_A_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_B_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_C[TILE_SIZE_2][TILE_SIZE_2];

    // Quantized accumulation for outer loop.
    float sum_outer;
    float value_outer;
    float c_outer = 0;
    float y_outer;
    float t_outer;

    // Loop over the tiles of the input in steps of TILE_SIZE_2
    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
        // Collaborative loading of tiles into shared memory
        const int input_col = t * TILE_SIZE_2 + threadIdx.x;
        const int weight_row = t * TILE_SIZE_2 + threadIdx.y;

        if (row < in_batch && input_col < in_features){
            shared_A[threadIdx.y][threadIdx.x] = input[prt * in_batch * in_features + row * in_features + input_col];
            shared_A_scale[threadIdx.y][threadIdx.x] = scale_input[prt * in_batch * in_features + row * in_features + input_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
            shared_A_scale[threadIdx.y][threadIdx.x] = 0;
        }
        if (col < out_features && weight_row < in_features){
            shared_B[threadIdx.y][threadIdx.x] = weight[prt * out_features * in_features + col * in_features + weight_row];
            shared_B_scale[threadIdx.y][threadIdx.x] = scale_weight[prt * out_features * in_features + col * in_features + weight_row];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
            shared_B_scale[threadIdx.y][threadIdx.x] = 0;
        }
        if (row < in_batch && col < out_features){
            shared_C[threadIdx.y][threadIdx.x] = output[prt * in_batch * out_features + row * out_features + col];
        } else {
            shared_C[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Quantized accumulation for inner loop.
        float acc = 0;
        float c_inner = 0;
        float y_inner;
        float t_inner;

        // Perform the multiplication for this tile
        for (int k=0; k < TILE_SIZE_2; ++k){
            float scaled_product = shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x] * 
                                   shared_A_scale[threadIdx.y][k] * shared_B_scale[k][threadIdx.x];

            scaled_product = round_rne_fp_full(scaled_product, man_width, exp_width);
            y_inner = round_rne_fp_full(scaled_product - c_inner, man_width, exp_width);
            t_inner = round_rne_fp_full(acc + y_inner, man_width, exp_width);
            c_inner = round_rne_fp_full(t_inner - acc, man_width, exp_width) - y_inner;
            c_inner = round_rne_fp_full(c_inner, man_width, exp_width);
            acc = round_rne_fp_full(t_inner, man_width, exp_width);
        }

        // Load sum and value for outer loop
        sum_outer = shared_C[threadIdx.y][threadIdx.x];
        value_outer = acc;

        y_outer = round_rne_fp_full(value_outer - c_outer, man_width, exp_width);
        t_outer = round_rne_fp_full(sum_outer + y_outer, man_width, exp_width);
        c_outer = round_rne_fp_full(t_outer - sum_outer, man_width, exp_width) - y_outer;
        c_outer = round_rne_fp_full(c_outer, man_width, exp_width);
        sum_outer = round_rne_fp_full(t_outer, man_width, exp_width);

        if (row < in_batch && col < out_features){
            output[prt * in_batch * out_features + row * out_features + col] = sum_outer;
        }

        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void ordmm_chunk_full_quant_bcast_scaled_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float* __restrict__ scale_input,
    const float* __restrict__ scale_weight,
    float* __restrict__ output,
    int in_batch, int in_features, int out_features,
    int man_width, int exp_width
){

    int col = blockIdx.x * TILE_SIZE_2 + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE_2 + threadIdx.y;
    int prt = blockIdx.z;

    // Shared memory for tiles of input and weight
    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_A_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_B_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_C[TILE_SIZE_2][TILE_SIZE_2];

    float acc;

    // Loop over the tiles of the input in steps of TILE_SIZE_2
    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
        // Collaborative loading of tiles into shared memory
        const int input_col = t * TILE_SIZE_2 + threadIdx.x;
        const int weight_row = t * TILE_SIZE_2 + threadIdx.y;

        if (row < in_batch && input_col < in_features){
            shared_A[threadIdx.y][threadIdx.x] = input[prt * in_batch * in_features + row * in_features + input_col];
            shared_A_scale[threadIdx.y][threadIdx.x] = scale_input[prt * in_batch * in_features + row * in_features + input_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
            shared_A_scale[threadIdx.y][threadIdx.x] = 0;
        }
        if (col < out_features && weight_row < in_features){
            shared_B[threadIdx.y][threadIdx.x] = weight[prt * out_features * in_features + col * in_features + weight_row];
            shared_B_scale[threadIdx.y][threadIdx.x] = scale_weight[prt * out_features * in_features + col * in_features + weight_row];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
            shared_B_scale[threadIdx.y][threadIdx.x] = 0;
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
            float scaled_product = shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x] * 
                                   shared_A_scale[threadIdx.y][k] * shared_B_scale[k][threadIdx.x];
            acc += scaled_product;
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

torch::Tensor ordmm_chunk_bcast_scaled(
    torch::Tensor input,
    torch::Tensor weight_tpose,
    torch::Tensor scale_input,
    torch::Tensor scale_weight_tpose,
    int man_width, int exp_width,
    std::string sum_type="quant"
){
    // Broadcast tensors to compatible batch shapes
    auto batch_shape = torch::infer_size(
        input.sizes().slice(0, input.dim() - 2),
        weight_tpose.sizes().slice(0, weight_tpose.dim() - 2)
    );
    std::vector<int64_t> input_expanded_shape = batch_shape;
    auto input_last_dims = input.sizes().slice(input.dim() - 2, 2);
    input_expanded_shape.insert(input_expanded_shape.end(), input_last_dims.begin(), input_last_dims.end());
    input = input.expand(input_expanded_shape);
    scale_input = scale_input.expand(input_expanded_shape);
    
    std::vector<int64_t> weight_tpose_expanded_shape = batch_shape;
    auto weight_tpose_last_dims = weight_tpose.sizes().slice(weight_tpose.dim() - 2, 2);
    weight_tpose_expanded_shape.insert(weight_tpose_expanded_shape.end(), weight_tpose_last_dims.begin(), weight_tpose_last_dims.end());
    weight_tpose = weight_tpose.expand(weight_tpose_expanded_shape);
    scale_weight_tpose = scale_weight_tpose.expand(weight_tpose_expanded_shape);
    
    int64_t batch_size = std::accumulate(batch_shape.begin(), batch_shape.end(), 1L, std::multiplies<int64_t>());
    auto input_flat = input.reshape({batch_size, input.size(-2), input.size(-1)}).to(weight_tpose.dtype());
    auto weight_tpose_flat = weight_tpose.reshape({batch_size, weight_tpose.size(-2), weight_tpose.size(-1)});
    auto scale_input_flat = scale_input.reshape({batch_size, scale_input.size(-2), scale_input.size(-1)});
    auto scale_weight_tpose_flat = scale_weight_tpose.reshape({batch_size, scale_weight_tpose.size(-2), scale_weight_tpose.size(-1)});

    std::vector<int64_t> target_shape = input.sizes().slice(0, input.sizes().size() - 1).vec();
    target_shape.push_back(weight_tpose.size(-2));

    input_flat = input_flat.contiguous();
    weight_tpose_flat = weight_tpose_flat.contiguous();
    scale_input_flat = scale_input_flat.contiguous().to(torch::kFloat);
    scale_weight_tpose_flat = scale_weight_tpose_flat.contiguous().to(torch::kFloat);

    int part = input_flat.size(0);
    int in_batch = input_flat.size(1);
    int in_features = input_flat.size(2);
    int out_features = weight_tpose_flat.size(1);

    torch::Tensor output = torch::zeros({part, in_batch, out_features}, torch::TensorOptions().dtype(torch::kFloat).device(input.device()));

    dim3 block_dim(TILE_SIZE_2, TILE_SIZE_2);
    dim3 grid_dim((out_features + TILE_SIZE_2 - 1) / TILE_SIZE_2, (in_batch + TILE_SIZE_2 - 1) / TILE_SIZE_2, part);

    if(sum_type == "quant"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "matmul_chunk_scaled_quant", ([&]{
            ordmm_chunk_full_quant_bcast_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                weight_tpose_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                scale_weight_tpose_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                in_batch,
                in_features,
                out_features,
                man_width,
                exp_width
            );
        }));
    }else if (sum_type == "kahan"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "matmul_chunk_scaled_kahan", ([&]{
            ordmm_chunk_comp_sum_bcast_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                weight_tpose_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                scale_weight_tpose_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                in_batch,
                in_features,
                out_features,
                man_width,
                exp_width
            );
        }));
    }else if (sum_type == "2sum"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "matmul_chunk_scaled_kahan", ([&]{
            ordmm_chunk_comp_sum_bcast_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                weight_tpose_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                scale_weight_tpose_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                in_batch,
                in_features,
                out_features,
                man_width,
                exp_width
            );
        }));
    }else if (sum_type == "fast2sum"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "matmul_chunk_scaled_kahan", ([&]{
            ordmm_chunk_comp_sum_bcast_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                weight_tpose_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                scale_weight_tpose_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                in_batch,
                in_features,
                out_features,
                man_width,
                exp_width
            );
        }));
    }else if (sum_type == "neumaier"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "matmul_chunk_scaled_kahan", ([&]{
            ordmm_chunk_comp_sum_bcast_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                weight_tpose_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                scale_weight_tpose_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                in_batch,
                in_features,
                out_features,
                man_width,
                exp_width
            );
        }));
    }else if (sum_type == "klein"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "matmul_chunk_scaled_kahan", ([&]{
            ordmm_chunk_comp_sum_bcast_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                weight_tpose_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                scale_weight_tpose_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                in_batch,
                in_features,
                out_features,
                man_width,
                exp_width
            );
        }));
    } else {
        throw std::invalid_argument("sum_type has an invalid value");
    }
    cudaDeviceSynchronize();

    return output.view(target_shape);
}


#endif // LINEAR_CUH_SCALED
