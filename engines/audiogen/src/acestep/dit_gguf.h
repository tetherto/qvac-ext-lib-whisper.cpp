#pragma once

// ACE-Step DiT — GGUF weight IO.
//
// mmaps the DiT GGUF (acestep.cpp convert.py) and uploads weights into
// pre-allocated backend tensors. Unlike the VAE (weight_norm fusion), the DiT
// loader:
//   - reads config from GGUF metadata (acestep-dit.* / acestep.*),
//   - optionally fuses Q/K/V (self+cross attn) and gate/up (MLP) into single
//     tensors when the source quant types match (matmul throughput),
//   - pre-permutes proj_in ([H,in_ch,P] -> [in_ch*P, H]) and proj_out
//     ([H,out_ch,P] -> [H, out_ch*P]) at load time to drop runtime permutes.
//
// Kept as a separate translation unit (mirrors vae_gguf) so the graph builder in
// dit_ggml.cpp never touches gguf directly.

#include "dit_ggml.h"  // DitConfig
#include "gguf_mmap.h"

#include "ggml.h"
#include "ggml-backend.h"  // ggml_backend_buffer_t (CPU map-in-place)
#include "gguf.h"

#include <cstdint>
#include <string>

namespace tts_cpp::acestep {

// Parsed GGUF: mmapped file + ggml metadata context (descriptors, no data).
struct DitGGUF {
    gguf_context * ctx      = nullptr;
    ggml_context * meta     = nullptr;
    MappedFile     file;
    size_t         data_off = 0;
};

bool          dit_gguf_open(DitGGUF & g, const std::string & path);
void          dit_gguf_close(DitGGUF & g);
const void *  dit_gdata(const DitGGUF & g, const std::string & name);  // raw mmap ptr, nullptr if absent
ggml_tensor * dit_gmeta(const DitGGUF & g, const std::string & name);  // metadata tensor, nullptr if absent
bool          dit_gguf_has(const DitGGUF & g, const std::string & name);

// --- CPU map-in-place (weights) --------------------------------------------
// Back verbatim (same-type) weight tensors DIRECTLY with the mmap'd GGUF bytes
// instead of allocating a ggml backend buffer and copying into it. On the CPU
// backend this turns the model's weights from dirty anonymous RAM (which iOS
// jetsam counts against the app) into clean, file-backed, evictable pages — the
// llama.cpp `use_mmap` strategy. The mmap in `g` MUST stay alive for the
// model's lifetime, so the caller keeps `g` (does not dit_gguf_close) until the
// model is freed. Only valid for host-memory backends (a GPU can't read a host
// pointer as device memory); callers gate this on
// ggml_backend_buft_is_host(ggml_backend_get_default_buffer_type(backend)) — a
// core ggml-backend predicate, unlike ggml_backend_is_cpu which lives in the
// (dynamically-loaded on Android/Linux-arm64) CPU backend module.
//
// Usage: create one shared buffer over the whole mmap, map each verbatim weight
// BEFORE ggml_backend_alloc_ctx_tensors (which then skips any tensor whose
// ->data is already set), and skip the corresponding upload/copy.
ggml_backend_buffer_t dit_gguf_cpu_map_buffer(const DitGGUF & g);

// Point `dst` at its bytes inside the mmap and attach `map_buf`. `dst` must have
// been created with the SAME type + shape as the GGUF tensor (create_like), so
// the mapped bytes are a byte-for-byte match for what a copy would have set.
// Returns false (and leaves `dst` untouched) if the tensor is absent.
bool dit_gguf_map_tensor(ggml_tensor * dst, const DitGGUF & g, const std::string & name,
                         ggml_backend_buffer_t map_buf);

// Metadata accessors (throw std::runtime_error on missing key).
uint32_t dit_gguf_u32(const DitGGUF & g, const std::string & key);
float    dit_gguf_f32(const DitGGUF & g, const std::string & key);
bool     dit_gguf_bool(const DitGGUF & g, const std::string & key, bool def);  // def if key missing

// Read the full DiT config from metadata. Returns false if any key is missing.
bool dit_gguf_read_config(const DitGGUF & g, DitConfig & cfg);

} // namespace tts_cpp::acestep
