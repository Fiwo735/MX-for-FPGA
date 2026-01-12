#ifndef SUM_CUH
#define SUM_CUH

#include <iostream>
#include <torch/extension.h>
#include "round_fp.cuh"

#define TILE_SIZE_SUM 256



template <typename scalar_t>
__global__ void ordacc_chunk_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    __shared__ float shared_data[TILE_SIZE_SUM];
    
    float acc = 0;
    
    // Process elements in chunks
    for (int t = 0; t < (reduce_dim + TILE_SIZE_SUM - 1) / TILE_SIZE_SUM; ++t){
        int elem_idx = t * TILE_SIZE_SUM + threadIdx.x;
        
        if (elem_idx < reduce_dim && threadIdx.x < TILE_SIZE_SUM){
            shared_data[threadIdx.x] = input[prt * batch_size * reduce_dim + idx * reduce_dim + elem_idx];
        } else {
            shared_data[threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        float local_acc = 0;
        
        // Sum elements in this chunk
        for (int k = 0; k < TILE_SIZE_SUM && (t * TILE_SIZE_SUM + k) < reduce_dim; ++k){
            local_acc += shared_data[k];
        }
        
        acc += local_acc;
        acc = round_rne_fp_full(acc, man_width, exp_width);
        
        __syncthreads();
    }
    
    output[prt * batch_size + idx] = acc;
}

template <typename scalar_t>
__global__ void ordacc_chunk_full_quant_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    __shared__ float shared_data[TILE_SIZE_SUM];
    
    float acc = 0;
    
    // Process elements in chunks
    for (int t = 0; t < (reduce_dim + TILE_SIZE_SUM - 1) / TILE_SIZE_SUM; ++t){
        int elem_idx = t * TILE_SIZE_SUM + threadIdx.x;
        
        if (elem_idx < reduce_dim && threadIdx.x < TILE_SIZE_SUM){
            shared_data[threadIdx.x] = input[prt * batch_size * reduce_dim + idx * reduce_dim + elem_idx];
        } else {
            shared_data[threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Sum elements in this chunk with quantization after each addition
        for (int k = 0; k < TILE_SIZE_SUM && (t * TILE_SIZE_SUM + k) < reduce_dim; ++k){
            acc += shared_data[k];
            acc = round_rne_fp_full(acc, man_width, exp_width);
        }
        
        __syncthreads();
    }
    
    output[prt * batch_size + idx] = acc;
}

template <typename scalar_t>
__global__ void ordacc_chunk_scaled_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale_input,
    float* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    __shared__ scalar_t shared_data[TILE_SIZE_SUM];
    __shared__ float shared_scale[TILE_SIZE_SUM];
    
    float acc = 0;
    
    // Process elements in chunks
    for (int t = 0; t < (reduce_dim + TILE_SIZE_SUM - 1) / TILE_SIZE_SUM; ++t){
        int elem_idx = t * TILE_SIZE_SUM + threadIdx.x;
        
        if (elem_idx < reduce_dim && threadIdx.x < TILE_SIZE_SUM){
            shared_data[threadIdx.x] = input[prt * batch_size * reduce_dim + idx * reduce_dim + elem_idx];
            shared_scale[threadIdx.x] = scale_input[prt * batch_size * reduce_dim + idx * reduce_dim + elem_idx];
        } else {
            shared_data[threadIdx.x] = 0;
            shared_scale[threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        float local_acc = 0;
        
        // Sum scaled elements in this chunk
        for (int k = 0; k < TILE_SIZE_SUM && (t * TILE_SIZE_SUM + k) < reduce_dim; ++k){
            local_acc += shared_data[k] * shared_scale[k];
        }
        
        acc += local_acc;
        acc = round_rne_fp_full(acc, man_width, exp_width);
        
        __syncthreads();
    }
    
    output[prt * batch_size + idx] = acc;
}

template <typename scalar_t>
__global__ void ordacc_chunk_full_quant_scaled_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale_input,
    float* __restrict__ output,
    int batch_size, int reduce_dim,
    int man_width, int exp_width
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prt = blockIdx.y;
    
    if (idx >= batch_size) return;
    
    __shared__ scalar_t shared_data[TILE_SIZE_SUM];
    __shared__ float shared_scale[TILE_SIZE_SUM];
    
    float acc = 0;
    
    // Process elements in chunks
    for (int t = 0; t < (reduce_dim + TILE_SIZE_SUM - 1) / TILE_SIZE_SUM; ++t){
        int elem_idx = t * TILE_SIZE_SUM + threadIdx.x;
        
        if (elem_idx < reduce_dim && threadIdx.x < TILE_SIZE_SUM){
            shared_data[threadIdx.x] = input[prt * batch_size * reduce_dim + idx * reduce_dim + elem_idx];
            shared_scale[threadIdx.x] = scale_input[prt * batch_size * reduce_dim + idx * reduce_dim + elem_idx];
        } else {
            shared_data[threadIdx.x] = 0;
            shared_scale[threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Sum scaled elements in this chunk with quantization after each addition
        for (int k = 0; k < TILE_SIZE_SUM && (t * TILE_SIZE_SUM + k) < reduce_dim; ++k){
            acc += shared_data[k] * shared_scale[k];
            acc = round_rne_fp_full(acc, man_width, exp_width);
        }
        
        __syncthreads();
    }
    
    output[prt * batch_size + idx] = acc;
}


torch::Tensor ordacc_chunk(
    torch::Tensor input,
    int man_width, int exp_width,
    bool full_quant=false
){
    // Input shape: [..., batch_size, reduce_dim]
    // Output shape: [..., batch_size]
    
    auto orig_dtype = input.dtype();
    std::vector<int64_t> output_shape = input.sizes().slice(0, input.sizes().size() - 1).vec();
    
    // Flatten all batch dimensions except the last two
    int64_t batch_size = 1;
    for (int i = 0; i < input.dim() - 2; ++i){
        batch_size *= input.size(i);
    }
    
    auto input_flat = input.reshape({batch_size, input.size(-2), input.size(-1)});
    input_flat = input_flat.contiguous();
    
    int part = input_flat.size(0);
    int rows = input_flat.size(1);
    int reduce_dim = input_flat.size(2);
    
    torch::Tensor output = torch::zeros({part, rows}, input.options());
    
    dim3 block_dim(TILE_SIZE_SUM);
    dim3 grid_dim((rows + TILE_SIZE_SUM - 1) / TILE_SIZE_SUM, part);
    
    if (!full_quant){
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
            input_flat.scalar_type(), "ordacc_chunk", ([&]{
            ordacc_chunk_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                rows,
                reduce_dim,
                man_width,
                exp_width
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
            input_flat.scalar_type(), "ordacc_chunk_full_quant", ([&]{
            ordacc_chunk_full_quant_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                rows,
                reduce_dim,
                man_width,
                exp_width
            );
        }));
    }
    
    cudaDeviceSynchronize();
    
    return output.view(output_shape).to(orig_dtype);
}

torch::Tensor ordacc_chunk_scaled(
    torch::Tensor input,
    torch::Tensor scale_input,
    int man_width, int exp_width,
    bool full_quant=false
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
    
    if (!full_quant){
        AT_DISPATCH_INTEGRAL_TYPES(input_flat.scalar_type(), "ordacc_chunk_scaled", ([&]{
            ordacc_chunk_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                rows,
                reduce_dim,
                man_width,
                exp_width
            );
        }));
    } else {
        AT_DISPATCH_INTEGRAL_TYPES(input_flat.scalar_type(), "ordacc_chunk_scaled_full_quant", ([&]{
            ordacc_chunk_full_quant_scaled_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input_flat.data_ptr<scalar_t>(),
                scale_input_flat.data_ptr<float>(),
                output.data_ptr<float>(),
                rows,
                reduce_dim,
                man_width,
                exp_width
            );
        }));
    }
    
    cudaDeviceSynchronize();
    
    return output.view(output_shape);
}

#endif // SUM_CUH
