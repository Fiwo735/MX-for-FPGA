#include <torch/extension.h>
#include "ordmm_chunk_bcast.cuh"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    // Software emulation of accumulator quantization.
    m.def("ordmm_chunk_bcast", &ordmm_chunk_bcast, "ordmm_chunk_bcast");
}
