#include <torch/extension.h>
#include "ordmm_chunk_bcast_scaled.cuh"
#include "ordacc_chunk.cuh"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    // Software emulation of accumulator quantization.
    m.def("ordmm_chunk_bcast_scaled", &ordmm_chunk_bcast_scaled, "ordmm_chunk_bcast_scaled");
    m.def("ordacc_chunk_scaled", &ordacc_chunk_scaled, "ordacc_chunk_scaled");
}
