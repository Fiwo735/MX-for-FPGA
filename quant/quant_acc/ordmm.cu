#include <torch/extension.h>
#include "ordmm_chunk_bcast_scaled.cuh"
#include "ordacc_chunk.cuh"
#include "ordacc_sum_int.cuh"
#include "ordmm_int.cuh"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    // Software emulation of accumulator quantization.
    m.def("ordmm_chunk_bcast_scaled", &ordmm_chunk_bcast_scaled, "ordmm_chunk_bcast_scaled");
    m.def("ordacc_chunk_scaled", &ordacc_chunk_scaled, "ordacc_chunk_scaled");
    m.def("ordmm_chunk_bcast_scaled_int", &ordmm_chunk_bcast_scaled_int, "ordmm_chunk_bcast_scaled_int");
    m.def("ordacc_chunk_scaled_int", &ordacc_chunk_scaled_int, "ordacc_chunk_scaled_int");
}
