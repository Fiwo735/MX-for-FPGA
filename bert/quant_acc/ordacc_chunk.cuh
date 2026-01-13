#ifndef SUM_CUH
#define SUM_CUH

#include <iostream>
#include <torch/extension.h>
#include "round_fp.cuh"

#define TILE_SIZE_SUM 256
#define ROUND_INTERVAL 32



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
    
    float acc = 0;
    int base_offset = prt * batch_size * reduce_dim + idx * reduce_dim;
    
    // Process elements and apply rounding every ROUND_INTERVAL elements
    for (int k = 0; k < reduce_dim; ++k){
        float val = static_cast<float>(input[base_offset + k]);
        float scale = scale_input[base_offset + k];
        acc += val * scale;
        
        // Apply rounding every 32 elements or at the end
        if ((k + 1) % ROUND_INTERVAL == 0 || k == reduce_dim - 1){
            acc = round_rne_fp_full(acc, man_width, exp_width);
        }
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
    
    float acc = 0;
    int base_offset = prt * batch_size * reduce_dim + idx * reduce_dim;
    
    // Apply rounding after every single addition
    for (int k = 0; k < reduce_dim; ++k){
        float val = static_cast<float>(input[base_offset + k]);
        float scale = scale_input[base_offset + k];
        acc += val * scale;
        acc = round_rne_fp_full(acc, man_width, exp_width);
    }
    
    output[prt * batch_size + idx] = acc;
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
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "ordacc_chunk_scaled", ([&]{
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
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_flat.scalar_type(), "ordacc_chunk_scaled_full_quant", ([&]{
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
