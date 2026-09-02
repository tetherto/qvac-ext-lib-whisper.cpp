#pragma once

// Byte totals collected by the metadata-only (`*_model_load_metadata_only`)
// stage loaders. A metadata-only load runs the real loader's backend
// resolution and tensor wiring but SIZES every allocation through
// ggml_backend_alloc_ctx_tensors_from_buft_size instead of performing it, and
// reads no tensor data. The returned model can build (and size) the real
// compute graphs, but must never have tensor data read or written, and must
// never be handed to a real forward/compute call.

#include <cstddef>

namespace tts_cpp::acestep {

struct AcestepStageMeasure {
    // Bytes ggml_backend_alloc_ctx_tensors would allocate on the stage's
    // backend buffer (the F32-widened/converted tensors, plus every weight on
    // non-host backends).
    size_t weights_alloc_bytes = 0;
    // On host (CPU) backends: bytes of quantised weights mapped in place off
    // the GGUF mmap (clean, file-backed pages). 0 on device backends.
    size_t weights_mapped_bytes = 0;
    // LM only: KV-cache buffer bytes (n_sets x n_layers x 2 x D x max_seq x Nkv, F16).
    size_t kv_bytes = 0;
};

}  // namespace tts_cpp::acestep
