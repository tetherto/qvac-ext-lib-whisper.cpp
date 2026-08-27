// acestep-quantize: GGUF requantizer for the ACE-Step stage files, ported from
// tools/quantize.cpp in acestep.cpp (MIT). Reads a BF16 GGUF produced by
// scripts/convert-acestep-to-gguf.py and writes a quantized GGUF using the
// mixed-precision policy in quantize_policy.h. Tensor data is streamed one
// tensor at a time so peak memory stays at the largest single tensor.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"

#include "gguf_mmap.h"
#include "quantize_policy.h"

namespace {

using tts_cpp::acestep::MappedFile;
using tts_cpp::acestep::QuantVariant;

constexpr const char * TAG = "acestep-quantize";

void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type>\n", prog);
    fprintf(stderr, "Types:");
    size_t               count    = 0;
    const QuantVariant * variants = tts_cpp::acestep::quant_variants(count);
    for (size_t i = 0; i < count; ++i) {
        fprintf(stderr, " %s", variants[i].name);
    }
    fprintf(stderr, "\n");
}

std::string read_arch(const gguf_context * inp) {
    const int64_t idx = gguf_find_key(inp, "general.architecture");
    if (idx < 0) {
        return "unknown";
    }
    return gguf_get_val_str(inp, idx);
}

int read_block_count(const gguf_context * inp, const std::string & arch) {
    const std::string key = arch + ".block_count";
    const int64_t     idx = gguf_find_key(inp, key.c_str());
    if (idx < 0) {
        return 0;
    }
    return (int) gguf_get_val_u32(inp, idx);
}

bool source_type_convertible(enum ggml_type type) {
    return type == GGML_TYPE_BF16 || type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

bool tensor_to_f32(const void * src, float * dst, int64_t n, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n);
            return true;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
            return true;
        case GGML_TYPE_F32:
            memcpy(dst, src, (size_t) n * sizeof(float));
            return true;
        default:
            return false;
    }
}

struct TensorPlan {
    enum ggml_type target;
    bool           quantize;
    bool           promote;
};

std::vector<TensorPlan> plan_tensors(gguf_context *       out,
                                     const gguf_context * inp,
                                     ggml_context *       meta,
                                     const std::string &  arch,
                                     const QuantVariant & variant,
                                     int                  n_layers) {
    const int               n_tensors = (int) gguf_get_n_tensors(inp);
    std::vector<TensorPlan> plans((size_t) n_tensors, { GGML_TYPE_COUNT, false, false });

    for (int i = 0; i < n_tensors; i++) {
        const char *  name   = gguf_get_tensor_name(inp, i);
        ggml_tensor * t      = ggml_get_tensor(meta, name);
        const int     n_dims = ggml_n_dims(t);

        gguf_add_tensor(out, t);

        const enum ggml_type target =
            tts_cpp::acestep::quant_pick_type(name, n_dims, arch.c_str(), variant, n_layers);

        if (target == GGML_TYPE_COUNT) {
            if (tts_cpp::acestep::quant_should_promote_f32(n_dims) &&
                (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16)) {
                gguf_set_tensor_type(out, name, GGML_TYPE_F32);
                plans[(size_t) i] = { GGML_TYPE_F32, false, true };
            }
            continue;
        }

        const bool aligned = t->ne[0] % ggml_blck_size(target) == 0;
        if (source_type_convertible(t->type) && aligned) {
            gguf_set_tensor_type(out, name, target);
            plans[(size_t) i] = { target, true, false };
        }
    }

    return plans;
}

size_t write_padding(FILE * fout, size_t data_pos, size_t alignment) {
    const size_t pad = (alignment - (data_pos % alignment)) % alignment;
    if (pad > 0) {
        const uint8_t zeros[64] = {};
        fwrite(zeros, 1, pad, fout);
    }
    return pad;
}

struct StreamStats {
    int     n_quantized = 0;
    int     n_promoted  = 0;
    int64_t bytes_in    = 0;
    int64_t bytes_out   = 0;
};

bool stream_tensor_data(FILE *                          fout,
                        const gguf_context *            inp,
                        ggml_context *                  meta,
                        const uint8_t *                 base,
                        size_t                          data_off,
                        size_t                          alignment,
                        const std::vector<TensorPlan> & plans,
                        StreamStats &                   stats) {
    const int n_tensors = (int) gguf_get_n_tensors(inp);
    size_t    data_pos  = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char *  name     = gguf_get_tensor_name(inp, i);
        ggml_tensor * t        = ggml_get_tensor(meta, name);
        const int64_t nel      = ggml_nelements(t);
        const size_t  src_size = ggml_nbytes(t);
        const void *  src      = base + data_off + gguf_get_tensor_offset(inp, i);

        stats.bytes_in += (int64_t) src_size;
        data_pos += write_padding(fout, data_pos, alignment);

        const TensorPlan & plan = plans[(size_t) i];

        if (plan.promote) {
            std::vector<float> f32((size_t) nel);
            if (!tensor_to_f32(src, f32.data(), nel, t->type)) {
                fprintf(stderr, "[%s] cannot promote %s from type %d\n", TAG, name, (int) t->type);
                return false;
            }
            const size_t out_size = (size_t) nel * sizeof(float);
            fwrite(f32.data(), 1, out_size, fout);
            data_pos += out_size;
            stats.bytes_out += (int64_t) out_size;
            stats.n_promoted++;
        } else if (plan.quantize) {
            std::vector<float> f32((size_t) nel);
            if (!tensor_to_f32(src, f32.data(), nel, t->type)) {
                fprintf(stderr, "[%s] cannot quantize %s from type %d\n", TAG, name, (int) t->type);
                return false;
            }
            const int64_t        n_per_row = t->ne[0];
            const int64_t        nrows     = nel / n_per_row;
            const size_t         qsize     = ggml_row_size(plan.target, n_per_row) * (size_t) nrows;
            std::vector<uint8_t> qbuf(qsize);
            ggml_quantize_chunk(plan.target, f32.data(), qbuf.data(), 0, nrows, n_per_row, nullptr);
            fwrite(qbuf.data(), 1, qsize, fout);
            data_pos += qsize;
            stats.bytes_out += (int64_t) qsize;
            stats.n_quantized++;
        } else {
            fwrite(src, 1, src_size, fout);
            data_pos += src_size;
            stats.bytes_out += (int64_t) src_size;
        }
    }

    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char *         inp_path = argv[1];
    const char *         out_path = argv[2];
    const QuantVariant * variant  = tts_cpp::acestep::find_quant_variant(argv[3]);

    if (!variant) {
        fprintf(stderr, "[%s] unknown type: %s\n", TAG, argv[3]);
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "[%s] %s -> %s (%s)\n", TAG, inp_path, out_path, variant->name);

    MappedFile mapping;
    if (!tts_cpp::acestep::mapped_file_open(mapping, inp_path, TAG)) {
        return 1;
    }

    ggml_context *   meta   = nullptr;
    gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&meta };
    gguf_context *   inp    = gguf_init_from_file(inp_path, params);
    if (!inp) {
        fprintf(stderr, "[%s] failed to read %s\n", TAG, inp_path);
        tts_cpp::acestep::mapped_file_close(mapping);
        return 1;
    }

    const std::string arch     = read_arch(inp);
    const int         n_layers = read_block_count(inp, arch);
    fprintf(stderr, "[%s] arch=%s layers=%d\n", TAG, arch.c_str(), n_layers);

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, inp);
    gguf_set_val_u32(out, "general.quantization_version", 2);
    gguf_set_val_str(out, "general.file_type", variant->name);

    const std::vector<TensorPlan> plans = plan_tensors(out, inp, meta, arch, *variant, n_layers);

    if (!gguf_write_to_file(out, out_path, /*only_meta=*/true)) {
        fprintf(stderr, "[%s] failed to write metadata to %s\n", TAG, out_path);
        return 1;
    }

    FILE * fout = fopen(out_path, "ab");
    if (!fout) {
        fprintf(stderr, "[%s] failed to open %s for append\n", TAG, out_path);
        return 1;
    }

    StreamStats stats;
    const bool  ok = stream_tensor_data(fout, inp, meta, mapping.data, gguf_get_data_offset(inp),
                                        gguf_get_alignment(out), plans, stats);
    fclose(fout);

    gguf_free(out);
    gguf_free(inp);
    ggml_free(meta);
    tts_cpp::acestep::mapped_file_close(mapping);

    if (!ok) {
        remove(out_path);
        return 1;
    }

    fprintf(stderr, "[%s] quantized %d/%d tensors, promoted %d to F32\n", TAG, stats.n_quantized,
            (int) plans.size(), stats.n_promoted);
    fprintf(stderr, "[%s] %.1f GB -> %.1f GB (%.1fx)\n", TAG, (double) stats.bytes_in / 1e9,
            (double) stats.bytes_out / 1e9,
            stats.bytes_out > 0 ? (double) stats.bytes_in / (double) stats.bytes_out : 0.0);
    fprintf(stderr, "[%s] wrote %s\n", TAG, out_path);
    return 0;
}
