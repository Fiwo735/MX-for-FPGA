#ifndef SUM_CUH
#define SUM_CUH

#include <iostream>
#include <torch/extension.h>
#include "round_fp.cuh"

#define TILE_SIZE_SUM 256
#define ROUND_INTERVAL 32



template <typename scalar_t>
__global__ void ordacc_chunk_comp_sum_scaled_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale_input,
    float* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width, int group_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    int base_offset = prt * batch_size * reduce_dim + idx * reduce_dim;
    
    float sum_outer = 0;
    float value_outer;
    float c_outer = 0;
    float y_outer;
    float t_outer;
    float sum_inner = 0;
    float value_inner;
    float c_inner = 0;
    float y_inner;
    float t_inner;
    
    // Apply rounding after every single addition
    for (int k = 0; k < reduce_dim; ++k){
        float val = static_cast<float>(input[base_offset + k]);
        float scaled_product = val;
        value_inner = scaled_product;

        y_inner = round_rne_fp_full(value_inner - c_inner, man_width, exp_width);
        t_inner = round_rne_fp_full(sum_inner + y_inner, man_width, exp_width);
        c_inner = round_rne_fp_full(t_inner - sum_inner, man_width, exp_width) - y_inner;
        c_inner = round_rne_fp_full(c_inner, man_width, exp_width);
        sum_inner = round_rne_fp_full(t_inner, man_width, exp_width);

        if ((k + 1) % ROUND_INTERVAL == 0 || k == reduce_dim - 1){
            value_outer = sum_inner;
            sum_inner = 0;
            c_inner = 0;

            sum_outer += value_outer * scale_input[base_offset + k];
        }
    }
    
    output[prt * batch_size + idx] = sum_outer;
}

template <typename scalar_t>
__global__ void ordacc_chunk_2sum_scaled_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale_input,
    float* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width, int group_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    int base_offset = prt * batch_size * reduce_dim + idx * reduce_dim;
    
    float sum_outer = 0;
    float value_outer;
    float error_outer = 0;
    float s_outer;
    float d_sum_outer;
    float d_value_outer;
    float sum_p_outer;
    float value_p_outer;
    float d_added_outer;
    float sum_inner = 0;
    float value_inner;
    float error_inner = 0;
    float s_inner;
    float d_sum_inner;
    float d_value_inner;
    float sum_p_inner;
    float value_p_inner;
    float d_added_inner;
    
    // Apply rounding after every single addition
    for (int k = 0; k < reduce_dim; ++k){
        float val = static_cast<float>(input[base_offset + k]);
        float scaled_product = val;
        value_inner = scaled_product;

        s_inner = round_rne_fp_full(sum_inner + value_inner, man_width, exp_width);
        sum_p_inner = round_rne_fp_full(s_inner - value_inner, man_width, exp_width);
        value_p_inner = round_rne_fp_full(s_inner - sum_p_inner, man_width, exp_width);
        d_sum_inner = round_rne_fp_full(sum_inner - sum_p_inner, man_width, exp_width);
        d_value_inner = round_rne_fp_full(value_inner - value_p_inner, man_width, exp_width);
        d_added_inner = round_rne_fp_full(d_sum_inner + d_value_inner, man_width, exp_width);
        error_inner = round_rne_fp_full(error_inner + d_added_inner, man_width, exp_width);
        sum_inner = s_inner;

        if ((k + 1) % ROUND_INTERVAL == 0 || k == reduce_dim - 1){
            value_outer = round_rne_fp_full(sum_inner + error_inner, man_width, exp_width);
            sum_inner = 0;
            error_inner = 0;

            sum_outer += value_outer * scale_input[base_offset + k];
        }
    }
    
    output[prt * batch_size + idx] = sum_outer;
}

template <typename scalar_t>
__global__ void ordacc_chunk_fast2sum_scaled_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale_input,
    float* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width, int group_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    int base_offset = prt * batch_size * reduce_dim + idx * reduce_dim;
    
    float sum_outer = 0;
    float value_outer;
    float error_outer = 0;
    float s_outer;
    float z_outer;
    float val_z_sub_outer;
    float sum_inner = 0;
    float value_inner;
    float error_inner = 0;
    float s_inner;
    float z_inner;
    float val_z_sub_inner;
    
    // Apply rounding after every single addition
    for (int k = 0; k < reduce_dim; ++k){
        float val = static_cast<float>(input[base_offset + k]);
        float scaled_product = val;
        value_inner = scaled_product;

        s_inner = round_rne_fp_full(sum_inner + value_inner, man_width, exp_width);
        z_inner = round_rne_fp_full(s_inner - sum_inner, man_width, exp_width);
        val_z_sub_inner = round_rne_fp_full(value_inner - z_inner, man_width, exp_width);
        error_inner = round_rne_fp_full(error_inner + val_z_sub_inner, man_width, exp_width);
        sum_inner = s_inner;

        if ((k + 1) % ROUND_INTERVAL == 0 || k == reduce_dim - 1){
            value_outer = round_rne_fp_full(sum_inner + error_inner, man_width, exp_width);
            sum_inner = 0;
            error_inner = 0;

            sum_outer += value_outer * scale_input[base_offset + k];
        }
    }
    
    output[prt * batch_size + idx] = sum_outer;
}

template <typename scalar_t>
__global__ void ordacc_chunk_neumaier_scaled_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale_input,
    float* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width, int group_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    int base_offset = prt * batch_size * reduce_dim + idx * reduce_dim;
    
    float sum_outer = 0;
    float value_outer;
    float c_outer = 0;
    float s_outer;
    float sum_inner = 0;
    float value_inner;
    float c_inner = 0;
    float s_inner;
    
    // Apply rounding after every single addition
    for (int k = 0; k < reduce_dim; ++k){
        float val = static_cast<float>(input[base_offset + k]);
        float scaled_product = val;
        value_inner = scaled_product;

        s_inner = round_rne_fp_full(sum_inner + value_inner, man_width, exp_width);
        c_inner += (fabsf(sum_inner) >= fabsf(value_inner)) ?
            round_rne_fp_full(round_rne_fp_full(sum_inner - s_inner, man_width, exp_width) + value_inner, man_width, exp_width):
            round_rne_fp_full(round_rne_fp_full(value_inner - s_inner, man_width, exp_width) + sum_inner, man_width, exp_width);
        c_inner = round_rne_fp_full(c_inner, man_width, exp_width);
        sum_inner = s_inner;

        if ((k + 1) % ROUND_INTERVAL == 0 || k == reduce_dim - 1){
            value_outer = round_rne_fp_full(sum_inner + c_inner, man_width, exp_width);
            sum_inner = 0;
            c_inner = 0;

            sum_outer += value_outer * scale_input[base_offset + k];
        }
    }
    
    output[prt * batch_size + idx] = sum_outer;
}

template <typename scalar_t>
__global__ void ordacc_chunk_klein_scaled_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale_input,
    float* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width, int group_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    int base_offset = prt * batch_size * reduce_dim + idx * reduce_dim;
    
    float sum_outer = 0;
    float value_outer;
    float cs_outer = 0;
    float ccs_outer = 0;
    float s_outer;
    float t_outer;
    float c_outer;
    float cc_outer;
    float sum_inner = 0;
    float value_inner;
    float cs_inner = 0;
    float ccs_inner = 0;
    float s_inner;
    float t_inner;
    float c_inner;
    float cc_inner;
    
    // Apply rounding after every single addition
    for (int k = 0; k < reduce_dim; ++k){
        float val = static_cast<float>(input[base_offset + k]);
        float scaled_product = val;
        value_inner = scaled_product;

        s_inner = round_rne_fp_full(sum_inner + value_inner, man_width, exp_width);
        c_inner = (fabsf(sum_inner) >= fabsf(value_inner)) ?
            round_rne_fp_full(round_rne_fp_full(sum_inner - s_inner, man_width, exp_width) + value_inner, man_width, exp_width):
            round_rne_fp_full(round_rne_fp_full(value_inner - s_inner, man_width, exp_width) + sum_inner, man_width, exp_width);
        sum_inner = s_inner;
        t_inner = round_rne_fp_full(cs_inner + c_inner, man_width, exp_width);
        cc_inner = (fabsf(cs_inner) >= fabsf(c_inner)) ?
            round_rne_fp_full(round_rne_fp_full(cs_inner - t_inner, man_width, exp_width) + c_inner, man_width, exp_width):
            round_rne_fp_full(round_rne_fp_full(c_inner - t_inner, man_width, exp_width) + cs_inner, man_width, exp_width);
        cs_inner = t_inner;
        ccs_inner = round_rne_fp_full(ccs_inner + cc_inner, man_width, exp_width);

        if ((k + 1) % ROUND_INTERVAL == 0 || k == reduce_dim - 1){
            value_outer = round_rne_fp_full(sum_inner + round_rne_fp_full(cs_inner + ccs_inner, man_width, exp_width), man_width, exp_width);
            sum_inner = 0;
            cs_inner = 0;
            ccs_inner = 0;

            sum_outer += value_outer * scale_input[base_offset + k];
        }
    }
    
    output[prt * batch_size + idx] = sum_outer;
}

template <typename scalar_t>
__global__ void ordacc_chunk_full_quant_scaled_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale_input,
    float* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width, int group_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;

    int base_offset = prt * batch_size * reduce_dim + idx * reduce_dim;

    float sum_outer = 0;
    float sum_inner = 0;

    for (int k = 0; k < reduce_dim; ++k){
        float val = static_cast<float>(input[base_offset + k]);
        float scaled_product = val;

        sum_inner = round_rne_fp_full(sum_inner + scaled_product, man_width, exp_width);

        if ((k + 1) % ROUND_INTERVAL == 0 || k == reduce_dim - 1){
            sum_outer += sum_inner * scale_input[base_offset + k];
            sum_inner = 0;
        }
    }
    
    output[prt * batch_size + idx] = sum_outer;
}



torch::Tensor ordacc_chunk_scaled(
    torch::Tensor input,
    torch::Tensor scale_input,
    int man_width, int exp_width, int group_size,
    std::string sum_type="quant"
){
    // Input shape: [..., batch_size, reduce_dim]
    // Output shape: [..., batch_size]
    
    std::vector<int64_t> output_shape = input.sizes().slice(0, input.sizes().size() - 1).vec();
    
    // Flatten all batch dimensions except the last two
    int64_t batch_size = 1;
    for (int i = 0; i < input.dim() - 2; ++i){
        batch_size *= input.size(i);
    }
    
    auto input_flat = input.reshape({batch_size, input.size(-2), input.size(-1)});
    auto scale_input_flat = scale_input.reshape({batch_size, scale_input.size(-2), scale_input.size(-1)});
    
    input_flat = input_flat.contiguous();
    scale_input_flat = scale_input_flat.contiguous().to(torch::kFloat);
    
    int part = input_flat.size(0);
    int rows = input_flat.size(1);
    int reduce_dim = input_flat.size(2);
    
    torch::Tensor output = torch::zeros({part, rows}, 
        torch::TensorOptions().dtype(torch::kFloat).device(input.device()));
    
    dim3 block_dim(TILE_SIZE_SUM);
    dim3 grid_dim((rows + TILE_SIZE_SUM - 1) / TILE_SIZE_SUM, part);
    
    if (sum_type == "QUANT"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "ordacc_chunk_scaled_quant", ([&]{
            ordacc_chunk_full_quant_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                rows,
                reduce_dim,
                man_width,
                exp_width,
                group_size
            );
        }));
    } else if (sum_type == "KAHAN"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "ordacc_chunk_scaled_kahan", ([&]{
            ordacc_chunk_comp_sum_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                rows,
                reduce_dim,
                man_width,
                exp_width,
                group_size
            );
        }));
    } else if (sum_type == "TWOSUM"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "ordacc_chunk_scaled_2sum", ([&]{
            ordacc_chunk_2sum_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                rows,
                reduce_dim,
                man_width,
                exp_width,
                group_size
            );
        }));
    } else if (sum_type == "FASTTWOSUM"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "ordacc_chunk_scaled_fast2sum", ([&]{
            ordacc_chunk_fast2sum_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                rows,
                reduce_dim,
                man_width,
                exp_width,
                group_size
            );
        }));
    } else if (sum_type == "NEUMAIER"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "ordacc_chunk_scaled_neumaier", ([&]{
            ordacc_chunk_neumaier_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                rows,
                reduce_dim,
                man_width,
                exp_width,
                group_size
            );
        }));
    } else if (sum_type == "KLEIN"){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "ordacc_chunk_scaled_klein", ([&]{
            ordacc_chunk_klein_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                rows,
                reduce_dim,
                man_width,
                exp_width,
                group_size
            );
        }));
    } else {
        throw std::invalid_argument("sum_type has an invalid value");
    }
    
    cudaDeviceSynchronize();
    
    return output.view(output_shape);
}

#endif // SUM_CUH
