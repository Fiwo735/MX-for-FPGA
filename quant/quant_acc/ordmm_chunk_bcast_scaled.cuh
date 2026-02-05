#ifndef LINEAR_CUH_SCALED
#define LINEAR_CUH_SCALED

#include <iostream>
#include <torch/extension.h>
#include "round_fp.cuh"



// Template-based kernel implementations for different tile sizes
template <typename scalar_t, int TILE_SIZE_2>
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

    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_A_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_B_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_C[TILE_SIZE_2][TILE_SIZE_2];

    float sum_outer;
    float value_outer;
    float c_outer = 0;
    float y_outer;
    float t_outer;

    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
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

        float acc = 0;
        float c_inner = 0;
        float y_inner;
        float t_inner;

        for (int k=0; k < TILE_SIZE_2; ++k){
            float scaled_product = shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];

            scaled_product = round_rne_fp_full(scaled_product, man_width, exp_width);
            y_inner = round_rne_fp_full(scaled_product - c_inner, man_width, exp_width);
            t_inner = round_rne_fp_full(acc + y_inner, man_width, exp_width);
            c_inner = round_rne_fp_full(t_inner - acc, man_width, exp_width) - y_inner;
            c_inner = round_rne_fp_full(c_inner, man_width, exp_width);
            acc = round_rne_fp_full(t_inner, man_width, exp_width);
        }

        sum_outer = shared_C[threadIdx.y][threadIdx.x];
        value_outer = acc;
        value_outer *= shared_A_scale[threadIdx.y][0] * shared_B_scale[0][threadIdx.x];

        if (row < in_batch && col < out_features){
            output[prt * in_batch * out_features + row * out_features + col] = sum_outer + value_outer;
        }

        __syncthreads();
    }
}

template <typename scalar_t, int TILE_SIZE_2>
__global__ void ordmm_chunk_2sum_bcast_scaled_kernel(
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

    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_A_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_B_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_C[TILE_SIZE_2][TILE_SIZE_2];

    float sum_outer;
    float value_outer;
    float error_outer = 0;
    float s_outer;
    float d_sum_outer;
    float d_value_outer;
    float sum_p_outer;
    float value_p_outer;
    float d_added_outer;

    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
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

        float acc = 0;
        float error_inner = 0;
        float s_inner;
        float d_sum_inner;
        float d_value_inner;
        float sum_p_inner;
        float value_p_inner;
        float d_added_inner;

        for (int k=0; k < TILE_SIZE_2; ++k){
            float scaled_product = shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];

            scaled_product = round_rne_fp_full(scaled_product, man_width, exp_width);
            s_inner = round_rne_fp_full(acc + scaled_product, man_width, exp_width);
            sum_p_inner = round_rne_fp_full(s_inner - scaled_product, man_width, exp_width);
            value_p_inner = round_rne_fp_full(s_inner - sum_p_inner, man_width, exp_width);
            d_sum_inner = round_rne_fp_full(acc - sum_p_inner, man_width, exp_width);
            d_value_inner = round_rne_fp_full(scaled_product - value_p_inner, man_width, exp_width);
            d_added_inner = round_rne_fp_full(d_sum_inner + d_value_inner, man_width, exp_width);
            error_inner = round_rne_fp_full(error_inner + d_added_inner, man_width, exp_width);
            acc = s_inner;
        }

        sum_outer = shared_C[threadIdx.y][threadIdx.x];
        value_outer = round_rne_fp_full(acc + error_inner, man_width, exp_width);
        value_outer *= shared_A_scale[threadIdx.y][0] * shared_B_scale[0][threadIdx.x];

        if (row < in_batch && col < out_features){
            output[prt * in_batch * out_features + row * out_features + col] = sum_outer + value_outer;
        }

        __syncthreads();
    }
}

template <typename scalar_t, int TILE_SIZE_2>
__global__ void ordmm_chunk_fast2sum_bcast_scaled_kernel(
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

    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_A_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_B_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_C[TILE_SIZE_2][TILE_SIZE_2];

    float sum_outer;
    float value_outer;
    float error_outer = 0;
    float s_outer;
    float z_outer;
    float val_z_sub_outer;

    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
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

        float acc = 0;
        float error_inner = 0;
        float s_inner;
        float z_inner;
        float val_z_sub_inner;

        for (int k=0; k < TILE_SIZE_2; ++k){
            float scaled_product = shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];

            scaled_product = round_rne_fp_full(scaled_product, man_width, exp_width);
            s_inner = round_rne_fp_full(acc + scaled_product, man_width, exp_width);
            z_inner = round_rne_fp_full(s_inner - acc, man_width, exp_width);
            val_z_sub_inner = round_rne_fp_full(scaled_product - z_inner, man_width, exp_width);
            error_inner = round_rne_fp_full(error_inner + val_z_sub_inner, man_width, exp_width);
            acc = s_inner;
        }

        sum_outer = shared_C[threadIdx.y][threadIdx.x];
        value_outer = round_rne_fp_full(acc + error_inner, man_width, exp_width);
        value_outer *= shared_A_scale[threadIdx.y][0] * shared_B_scale[0][threadIdx.x];

        if (row < in_batch && col < out_features){
            output[prt * in_batch * out_features + row * out_features + col] = sum_outer + value_outer;
        }

        __syncthreads();
    }
}

template <typename scalar_t, int TILE_SIZE_2>
__global__ void ordmm_chunk_neumaier_bcast_scaled_kernel(
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

    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_A_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_B_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_C[TILE_SIZE_2][TILE_SIZE_2];

    float sum_outer;
    float value_outer;
    float c_outer = 0;
    float s_outer;

    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
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

        float acc = 0;
        float c_inner = 0;
        float s_inner;

        for (int k=0; k < TILE_SIZE_2; ++k){
            float scaled_product = shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];

            scaled_product = round_rne_fp_full(scaled_product, man_width, exp_width);
            s_inner = round_rne_fp_full(acc + scaled_product, man_width, exp_width);
            c_inner += (fabsf(acc) >= fabsf(scaled_product)) ?
                round_rne_fp_full(round_rne_fp_full(acc - s_inner, man_width, exp_width) + scaled_product, man_width, exp_width):
                round_rne_fp_full(round_rne_fp_full(scaled_product - s_inner, man_width, exp_width) + acc, man_width, exp_width);
            c_inner = round_rne_fp_full(c_inner, man_width, exp_width);
            acc = s_inner;
        }

        sum_outer = shared_C[threadIdx.y][threadIdx.x];
        value_outer = round_rne_fp_full(acc + c_inner, man_width, exp_width);
        value_outer *= shared_A_scale[threadIdx.y][0] * shared_B_scale[0][threadIdx.x];

        if (row < in_batch && col < out_features){
            output[prt * in_batch * out_features + row * out_features + col] = sum_outer + value_outer;
        }

        __syncthreads();
    }
}

template <typename scalar_t, int TILE_SIZE_2>
__global__ void ordmm_chunk_klein_bcast_scaled_kernel(
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

    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_A_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_B_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_C[TILE_SIZE_2][TILE_SIZE_2];

    float sum_outer;
    float value_outer;
    float cs_outer = 0;
    float ccs_outer = 0;
    float s_outer;
    float t_outer;
    float c_outer;
    float cc_outer;

    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
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

        float acc = 0;
        float cs_inner = 0;
        float ccs_inner = 0;
        float s_inner;
        float t_inner;
        float c_inner;
        float cc_inner;

        for (int k=0; k < TILE_SIZE_2; ++k){
            float scaled_product = shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];

            scaled_product = round_rne_fp_full(scaled_product, man_width, exp_width);
            s_inner = round_rne_fp_full(acc + scaled_product, man_width, exp_width);
            c_inner = (fabsf(acc) >= fabsf(scaled_product)) ?
                round_rne_fp_full(round_rne_fp_full(acc - s_inner, man_width, exp_width) + scaled_product, man_width, exp_width):
                round_rne_fp_full(round_rne_fp_full(scaled_product - s_inner, man_width, exp_width) + acc, man_width, exp_width);
            acc = s_inner;
            t_inner = round_rne_fp_full(cs_inner + c_inner, man_width, exp_width);
            cc_inner = (fabsf(cs_inner) >= fabsf(c_inner)) ?
                round_rne_fp_full(round_rne_fp_full(cs_inner - t_inner, man_width, exp_width) + c_inner, man_width, exp_width):
                round_rne_fp_full(round_rne_fp_full(c_inner - t_inner, man_width, exp_width) + cs_inner, man_width, exp_width);
            cs_inner = t_inner;
            ccs_inner = round_rne_fp_full(ccs_inner + cc_inner, man_width, exp_width);
        }

        sum_outer = shared_C[threadIdx.y][threadIdx.x];
        value_outer = round_rne_fp_full(acc + round_rne_fp_full(cs_inner + ccs_inner, man_width, exp_width), man_width, exp_width);
        value_outer *= shared_A_scale[threadIdx.y][0] * shared_B_scale[0][threadIdx.x];

        if (row < in_batch && col < out_features){
            output[prt * in_batch * out_features + row * out_features + col] = sum_outer + value_outer;
        }

        __syncthreads();
    }
}

template <typename scalar_t, int TILE_SIZE_2>
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

    __shared__ scalar_t shared_A[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ scalar_t shared_B[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_A_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_B_scale[TILE_SIZE_2][TILE_SIZE_2];
    __shared__ float shared_C[TILE_SIZE_2][TILE_SIZE_2];

    float acc;

    for (int t=0; t < (in_features + TILE_SIZE_2 - 1) / TILE_SIZE_2; ++t){
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

        for (int k=0; k < TILE_SIZE_2; ++k){
            float scaled_product = shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            acc += scaled_product;
            acc = round_rne_fp_full(acc, man_width, exp_width);
        }
        acc *= shared_A_scale[threadIdx.y][0] * shared_B_scale[0][threadIdx.x];

        acc += shared_C[threadIdx.y][threadIdx.x];
        if (row < in_batch && col < out_features){
            output[prt * in_batch * out_features + row * out_features + col] = acc;
        }

        __syncthreads();
    }
}

// Kernel launcher helper template
template<typename scalar_t, int TILE_SIZE_2>
void launch_kernel(
    const std::string& sum_type,
    dim3 grid_dim, dim3 block_dim,
    const scalar_t* input, const scalar_t* weight,
    const float* scale_input, const float* scale_weight,
    float* output,
    int in_batch, int in_features, int out_features,
    int man_width, int exp_width
){
    if(sum_type == "QUANT"){
        ordmm_chunk_full_quant_bcast_scaled_kernel<scalar_t, TILE_SIZE_2><<<grid_dim, block_dim>>>(
            input, weight, scale_input, scale_weight, output,
            in_batch, in_features, out_features, man_width, exp_width);
    }else if(sum_type == "KAHAN"){
        ordmm_chunk_comp_sum_bcast_scaled_kernel<scalar_t, TILE_SIZE_2><<<grid_dim, block_dim>>>(
            input, weight, scale_input, scale_weight, output,
            in_batch, in_features, out_features, man_width, exp_width);
    }else if(sum_type == "TWOSUM"){
        ordmm_chunk_2sum_bcast_scaled_kernel<scalar_t, TILE_SIZE_2><<<grid_dim, block_dim>>>(
            input, weight, scale_input, scale_weight, output,
            in_batch, in_features, out_features, man_width, exp_width);
    }else if(sum_type == "FASTTWOSUM"){
        ordmm_chunk_fast2sum_bcast_scaled_kernel<scalar_t, TILE_SIZE_2><<<grid_dim, block_dim>>>(
            input, weight, scale_input, scale_weight, output,
            in_batch, in_features, out_features, man_width, exp_width);
    }else if(sum_type == "NEUMAIER"){
        ordmm_chunk_neumaier_bcast_scaled_kernel<scalar_t, TILE_SIZE_2><<<grid_dim, block_dim>>>(
            input, weight, scale_input, scale_weight, output,
            in_batch, in_features, out_features, man_width, exp_width);
    }else if(sum_type == "KLEIN"){
        ordmm_chunk_klein_bcast_scaled_kernel<scalar_t, TILE_SIZE_2><<<grid_dim, block_dim>>>(
            input, weight, scale_input, scale_weight, output,
            in_batch, in_features, out_features, man_width, exp_width);
    }
}

// Runtime tile size dispatcher
template<typename scalar_t>
void dispatch_tile_size(
    int tile_size,
    const std::string& sum_type,
    dim3 grid_dim, dim3 block_dim,
    const scalar_t* input, const scalar_t* weight,
    const float* scale_input, const float* scale_weight,
    float* output,
    int in_batch, int in_features, int out_features,
    int man_width, int exp_width
){
    switch(tile_size){
        case 2:
            launch_kernel<scalar_t, 2>(sum_type, grid_dim, block_dim, input, weight,
                scale_input, scale_weight, output, in_batch, in_features, out_features,
                man_width, exp_width);
            break;
        case 4:
            launch_kernel<scalar_t, 4>(sum_type, grid_dim, block_dim, input, weight,
                scale_input, scale_weight, output, in_batch, in_features, out_features,
                man_width, exp_width);
            break;
        case 8:
            launch_kernel<scalar_t, 8>(sum_type, grid_dim, block_dim, input, weight,
                scale_input, scale_weight, output, in_batch, in_features, out_features,
                man_width, exp_width);
            break;
        case 16:
            launch_kernel<scalar_t, 16>(sum_type, grid_dim, block_dim, input, weight,
                scale_input, scale_weight, output, in_batch, in_features, out_features,
                man_width, exp_width);
            break;
        case 32:
            launch_kernel<scalar_t, 32>(sum_type, grid_dim, block_dim, input, weight,
                scale_input, scale_weight, output, in_batch, in_features, out_features,
                man_width, exp_width);
            break;
        default:
            throw std::invalid_argument("tile_size must be one of {2, 4, 8, 16, 32}");
    }
}

torch::Tensor ordmm_chunk_bcast_scaled(
    torch::Tensor input,
    torch::Tensor weight_tpose,
    torch::Tensor scale_input,
    torch::Tensor scale_weight_tpose,
    int man_width, int exp_width, int tile_size=32,
    std::string sum_type="quant"
){
    // Validate tile_size
    if(tile_size != 2 && tile_size != 4 && tile_size != 8 && 
       tile_size != 16 && tile_size != 32){
        throw std::invalid_argument("tile_size must be one of {2, 4, 8, 16, 32}");
    }

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

    torch::Tensor output = torch::zeros({part, in_batch, out_features}, 
        torch::TensorOptions().dtype(torch::kFloat).device(input.device()));

    dim3 block_dim(tile_size, tile_size);
    dim3 grid_dim((out_features + tile_size - 1) / tile_size, 
                  (in_batch + tile_size - 1) / tile_size, part);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        input_flat.scalar_type(), "matmul_chunk_scaled", ([&]{
        dispatch_tile_size<scalar_t>(
            tile_size, sum_type, grid_dim, block_dim,
            input_flat.data_ptr<scalar_t>(),
            weight_tpose_flat.data_ptr<scalar_t>(),
            scale_input_flat.data_ptr<float>(),
            scale_weight_tpose_flat.data_ptr<float>(),
            output.data_ptr<float>(),
            in_batch, in_features, out_features,
            man_width, exp_width
        );
    }));

    cudaDeviceSynchronize();

    return output.view(target_shape);
}


#endif // LINEAR_CUH_SCALED
