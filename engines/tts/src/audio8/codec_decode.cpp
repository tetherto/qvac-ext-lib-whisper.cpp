// Audio8 codec synthesis: ten codebook rows per frame in, waveform out.
//
// The path mirrors the reference exactly -- look each code up, project it to
// the latent width and sum, run the windowed post transformer, upsample by
// four, then the DAC stack's four transposed-conv stages and a tanh.
//
// It runs as two graphs rather than one, because the two halves want opposite
// things. The post transformer is eight windowed layers reaching 128 frames
// back each, so between them they see roughly a thousand frames and the whole
// sequence has to go through at once -- cheap, since it works at one column
// per frame. The synthesis stack then expands each of those columns into 2048
// samples, which is where the memory goes, and it is causal with a context of
// only a dozen frames, so it runs in blocks sized to a memory budget. That
// keeps the scratch bounded instead of the 1.5 GB a whole 24 s take would need.

#include "audio8/codec_ops.h"
#include "audio8/graph.h"

#include "fit_price.h"
#include "fit_util.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {

namespace {
constexpr double SCRATCH_BUDGET_SHARE = 0.25;
constexpr size_t SCRATCH_BUDGET_CAP = 384u * 1024 * 1024;
// The narrowest block that still produces audio, so the floor of every search
// and the width taken when even it is over budget.
constexpr int MIN_BLOCK_FRAMES = 1;
}  // namespace

size_t synthesis_scratch_budget(size_t configured, size_t free_bytes, size_t total_bytes) {
    if (configured > 0) return configured;
    // A zero total is a backend saying it cannot tell rather than a device with
    // no memory: ggml-metal reports zero for both figures on the OS versions
    // predating the working-set query. Real pressure shows as a small free
    // against a nonzero total, and that does have to narrow the blocks.
    if (total_bytes == 0) return SCRATCH_BUDGET_CAP;
    const size_t share = static_cast<size_t>(free_bytes * SCRATCH_BUDGET_SHARE);
    return std::min(share, SCRATCH_BUDGET_CAP);
}

namespace {

const char * const CODE_INPUT = "codes";
const char * const MASK_INPUT = "post_mask";
const char * const POST_INPUT = "post";

struct latent_graph {
    ggml_tensor * semantic = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * post = nullptr;
};

struct synthesis_graph {
    ggml_tensor * latent = nullptr;
    ggml_tensor * pcm = nullptr;
};

ggml_tensor * code_row(ggml_context * ctx, ggml_tensor * codes, int row) {
    return ggml_view_1d(ctx, codes, codes->ne[0], static_cast<size_t>(row) * codes->nb[1]);
}

ggml_tensor * reconstruct(ggml_context * ctx, const quantizer_weights & quantizer,
                          ggml_tensor * ids) {
    return project(ctx, quantizer.out_proj, ggml_get_rows(ctx, quantizer.codebook, ids));
}

ggml_tensor * sum_quantizers(ggml_context * ctx, const std::vector<quantizer_weights> & bank,
                             ggml_tensor * codes, int first_row) {
    ggml_tensor * total = nullptr;
    for (size_t index = 0; index < bank.size(); ++index) {
        ggml_tensor * part = reconstruct(
            ctx, bank[index], code_row(ctx, codes, first_row + static_cast<int>(index)));
        total = total ? ggml_add(ctx, total, part) : part;
    }
    return total;
}

ggml_tensor * run_upsample(ggml_context * ctx, const std::vector<resample_stage> & stages,
                           ggml_tensor * signal, float eps) {
    for (const resample_stage & stage : stages) {
        signal = causal_conv_transpose(ctx, stage.conv, signal, stage.stride);
        signal = convnext_block(ctx, stage.convnext, signal, eps);
    }
    return signal;
}

ggml_tensor * run_decoder_stack(ggml_context * ctx, const codec_model & model,
                                ggml_tensor * signal) {
    const codec_hparams & hp = model.hp;
    signal = causal_conv(ctx, model.dec_in, signal, /*stride=*/1, /*dilation=*/1);
    for (const dac_stage & stage : model.dec_stages) {
        signal = snake(ctx, signal, stage.alpha, hp.snake_epsilon);
        signal = causal_conv_transpose(ctx, stage.conv, signal, stage.stride);
        signal = residual_stack(ctx, stage.units, hp.residual_dilations, signal,
                                hp.snake_epsilon);
    }
    signal = snake(ctx, signal, model.dec_out_alpha, hp.snake_epsilon);
    return ggml_tanh(ctx, causal_conv(ctx, model.dec_out, signal, /*stride=*/1,
                                      /*dilation=*/1));
}

int upsample_factor(const codec_model & model) {
    int factor = 1;
    for (const resample_stage & stage : model.upsample) factor *= stage.stride;
    return factor;
}

// The frames of history one block has to be handed before its own, walked back
// from a single output sample through the whole synthesis stack.
int synthesis_context(const codec_model & model) {
    int span = span_through_conv(1, conv_taps(model.dec_out), 1, 1);
    for (size_t index = model.dec_stages.size(); index-- > 0;) {
        const dac_stage & stage = model.dec_stages[index];
        span = span_through_units(span, stage.units, model.hp.residual_dilations);
        span = span_through_transpose(span, conv_taps(stage.conv), stage.stride);
    }
    span = span_through_conv(span, conv_taps(model.dec_in), 1, 1);
    for (size_t index = model.upsample.size(); index-- > 0;) {
        span = span_through_convnext(span, model.upsample[index].convnext);
        span = span_through_transpose(span, conv_taps(model.upsample[index].conv),
                                      model.upsample[index].stride);
    }
    return span - 1;
}

latent_graph build_latents(ggml_context * ctx, const codec_model & model, int n_frames) {
    ggml_tensor * codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_frames,
                                             model.hp.num_codebooks);
    ggml_set_name(codes, CODE_INPUT);
    ggml_set_input(codes);
    ggml_tensor * mask = input_f32(ctx, MASK_INPUT, n_frames, n_frames);

    latent_graph built;
    built.semantic = sum_quantizers(ctx, model.semantic_quantizers, codes, 0);
    built.residual = sum_quantizers(ctx, model.residual_quantizers, codes, 1);
    built.post =
        window_forward(ctx, model.post, ggml_add(ctx, built.semantic, built.residual),
                       mask, model.precise_outputs);
    return built;
}

synthesis_graph build_synthesis(ggml_context * ctx, const codec_model & model,
                                int n_frames) {
    ggml_tensor * post = input_f32(ctx, POST_INPUT, model.hp.latent_dim, n_frames);
    synthesis_graph built;
    built.latent = run_upsample(ctx, model.upsample, post, model.hp.convnext_norm_eps);
    built.pcm = run_decoder_stack(ctx, model, built.latent);
    return built;
}

int row_limit(const codec_hparams & hp, int row) {
    return (row == 0 ? hp.semantic_codebook_size : hp.residual_codebook_size) - 1;
}

void clamp_row(const int32_t * codes, int n_frames, int limit, int32_t * out) {
    for (int frame = 0; frame < n_frames; ++frame) {
        out[frame] = std::min(std::max(codes[frame], 0), limit);
    }
}

// Codes outside a codebook would index past the embedding table, so they are
// clamped the way the reference clamps them before the lookup.
std::vector<int32_t> clamped_codes(const codec_hparams & hp, const int32_t * codes,
                                   int n_frames) {
    std::vector<int32_t> out(static_cast<size_t>(hp.num_codebooks) * n_frames);
    for (int row = 0; row < hp.num_codebooks; ++row) {
        const size_t at = static_cast<size_t>(row) * n_frames;
        clamp_row(codes + at, n_frames, row_limit(hp, row), out.data() + at);
    }
    return out;
}

bool run_latents(codec_model & model, const int32_t * codes, int n_frames, int n_threads,
                 std::vector<float> & post, decode_taps * taps, std::string * error) {
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) {
        if (error) *error = "audio8: failed to create the latent graph context";
        return false;
    }
    const latent_graph built = build_latents(work.ctx, model, n_frames);
    mark_output(work.graph, built.post);
    if (taps) {
        mark_output(work.graph, built.semantic);
        mark_output(work.graph, built.residual);
    }
    bool use_sched = false;
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, model.allocr,
                       work.graph, "latent", use_sched, error)) {
        return false;
    }

    const std::vector<int32_t> clamped = clamped_codes(model.hp, codes, n_frames);
    write_input(work.graph, CODE_INPUT, clamped.data(), clamped.size() * sizeof(int32_t));
    const std::vector<float> mask = window_mask(model.post.spec, n_frames);
    write_input(work.graph, MASK_INPUT, mask.data(), mask.size() * sizeof(float));
    if (!compute_graph(model.backend, model.sched, work.graph, use_sched, n_threads,
                       "latent", error)) {
        return false;
    }

    read_output(built.post, post);
    if (taps) {
        read_output(built.semantic, taps->semantic);
        read_output(built.residual, taps->residual);
        taps->post = post;
    }
    return true;
}

// A block covers post-transformer columns [first - context, first + count) and
// keeps only what belongs to [first, first + count): the context is there to
// give the causal convolutions real history instead of the zeros they would
// otherwise pad with.
struct block {
    int begin = 0;
    int count = 0;
    int dropped = 0;
};

block block_at(int first, int n_frames, int context, int block_frames) {
    block span;
    span.begin = std::max(0, first - context);
    span.dropped = first - span.begin;
    span.count = std::min(block_frames, n_frames - first) + span.dropped;
    return span;
}

size_t device_scratch_budget(const codec_model & model) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(model.backend), &free_bytes,
                            &total_bytes);
    return synthesis_scratch_budget(model.synthesis_scratch_budget, free_bytes,
                                    total_bytes);
}

size_t block_scratch(codec_model & model, ggml_gallocr_t pricer, int columns,
                     bool with_taps) {
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) return SIZE_MAX;
    const synthesis_graph built = build_synthesis(work.ctx, model, columns);
    mark_output(work.graph, built.pcm);
    if (with_taps) mark_output(work.graph, built.latent);
    size_t size = 0;
    ggml_gallocr_reserve_n_size(pricer, work.graph, nullptr, nullptr, &size);
    return size;
}

int span_of(int block_frames, int n_frames, int context) {
    return std::min(block_frames + context, n_frames);
}

int widest_block(codec_model & model, ggml_gallocr_t pricer, int context, int n_frames,
                 bool with_taps, size_t budget) {
    int narrowest = MIN_BLOCK_FRAMES;
    int widest = n_frames;
    while (narrowest < widest) {
        const int mid = narrowest + (widest - narrowest + 1) / 2;
        const size_t cost =
            block_scratch(model, pricer, span_of(mid, n_frames, context), with_taps);
        if (cost <= budget) {
            narrowest = mid;
        } else {
            widest = mid - 1;
        }
    }
    return narrowest;
}

struct block_plan {
    int frames = MIN_BLOCK_FRAMES;
    size_t scratch = 0;
};

block_plan plan_blocks(codec_model & model, int context, int n_frames, bool with_taps) {
    ggml_gallocr_t pricer =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    block_plan plan;
    plan.frames = model.synthesis_block_frames > 0
                      ? std::min(model.synthesis_block_frames, n_frames)
                      : widest_block(model, pricer, context, n_frames, with_taps,
                                     device_scratch_budget(model));
    plan.scratch =
        block_scratch(model, pricer, span_of(plan.frames, n_frames, context), with_taps);
    ggml_gallocr_free(pricer);
    return plan;
}

void append_tail(const std::vector<float> & produced, int dropped, int per_column,
                 std::vector<float> & out) {
    const size_t skip = static_cast<size_t>(dropped) * per_column;
    out.insert(out.end(), produced.begin() + skip, produced.end());
}

bool run_block(codec_model & model, const std::vector<float> & post, const block & span,
               int n_threads, std::vector<float> & pcm_out, decode_taps * taps,
               std::string * error) {
    const codec_hparams & hp = model.hp;
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) {
        if (error) *error = "audio8: failed to create the synthesis graph context";
        return false;
    }
    const synthesis_graph built = build_synthesis(work.ctx, model, span.count);
    mark_output(work.graph, built.pcm);
    if (taps) mark_output(work.graph, built.latent);
    bool use_sched = false;
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, model.block_allocr,
                       work.graph, "synthesis", use_sched, error)) {
        return false;
    }

    const size_t offset = static_cast<size_t>(span.begin) * hp.latent_dim;
    write_input(work.graph, POST_INPUT, post.data() + offset,
                static_cast<size_t>(span.count) * hp.latent_dim * sizeof(float));
    if (!compute_graph(model.backend, model.sched, work.graph, use_sched, n_threads,
                       "synthesis", error)) {
        return false;
    }

    std::vector<float> produced;
    read_output(built.pcm, produced);
    append_tail(produced, span.dropped, hp.frame_size, pcm_out);
    if (taps) {
        read_output(built.latent, produced);
        append_tail(produced, span.dropped * upsample_factor(model), hp.latent_dim,
                    taps->latent);
    }
    return true;
}

bool run_synthesis_blocks(codec_model & model, const std::vector<float> & post,
                          int n_frames, int n_threads, const cancel_hook & cancel,
                          std::vector<float> & pcm_out, decode_taps * taps,
                          decode_timing & clock, std::string * error) {
    pcm_out.reserve(static_cast<size_t>(n_frames) * model.hp.frame_size);
    if (taps) taps->latent.clear();
    const int context = synthesis_context(model);
    const block_plan plan = plan_blocks(model, context, n_frames, taps != nullptr);
    const int block_frames = plan.frames;
    clock.block_frames = plan.frames;
    clock.block_scratch = plan.scratch;
    for (int first = 0; first < n_frames; first += block_frames) {
        if (cancelled(cancel, error)) return false;
        if (!run_block(model, post, block_at(first, n_frames, context, block_frames),
                       n_threads, pcm_out, taps, error)) {
            return false;
        }
    }
    return true;
}

bool check_decodable(const codec_model & model, int n_frames, std::string * error) {
    if (!model.has_decoder) {
        if (error) *error = "audio8: this codec GGUF has no decoder";
        return false;
    }
    if (n_frames > model.hp.max_frames) {
        if (error) {
            *error = "audio8: " + std::to_string(n_frames) +
                     " frames exceeds the baked RoPE table of " +
                     std::to_string(model.hp.max_frames);
        }
        return false;
    }
    return true;
}

}  // namespace

bool decode_codes(codec_model & model, const int32_t * codes, int n_frames,
                  int n_threads, const cancel_hook & cancel,
                  std::vector<float> & pcm_out, std::string * error,
                  decode_taps * taps, decode_timing * timing) {
    pcm_out.clear();
    if (!check_decodable(model, n_frames, error)) return false;
    if (n_frames <= 0) return true;

    decode_timing discarded;
    decode_timing & clock = timing ? *timing : discarded;

    std::vector<float> post;
    {
        stage_timer measure(clock.latent_ms);
        if (!run_latents(model, codes, n_frames, n_threads, post, taps, error)) return false;
    }
    stage_timer measure(clock.synthesis_ms);
    return run_synthesis_blocks(model, post, n_frames, n_threads, cancel, pcm_out, taps,
                                clock, error);
}

// Fit measurement (include/tts-cpp/audio8/fit.h): price what one decode_codes
// leaves resident, without allocating. The latent graph is the full-sequence
// pass run_latents makes; the synthesis block is the width the runtime's own
// planner (plan_blocks, itself already size-only) would settle on against the
// device memory available right now, priced through the same dual-path
// dispatch prepare_graph allocates with. The two arenas (model.allocr /
// model.block_allocr) are separate and both stay resident, so they sum.
bool measure_decode_memory(codec_model & model, int n_frames, codec_fit_measure & out) {
    using ::tts_cpp::fitutil::sat_add;
    out = codec_fit_measure{};
    if (!model.has_decoder || n_frames <= 0) return false;

    {
        scratch work(AUDIO8_MAX_NODES);
        if (!work.ok()) return false;
        const latent_graph built = build_latents(work.ctx, model, n_frames);
        mark_output(work.graph, built.post);
        ::tts_cpp::detail::fit_graph_price price;
        if (!fit_price_graph(model.backend, work.graph, 2 * AUDIO8_MAX_NODES, price)) {
            return false;
        }
        out.device_bytes = sat_add(out.device_bytes, price.device_bytes);
        out.host_bytes   = sat_add(out.host_bytes, price.host_bytes);
    }

    const int context = synthesis_context(model);
    const block_plan plan = plan_blocks(model, context, n_frames, /*with_taps=*/false);
    out.block_frames = plan.frames;
    out.block_span   = span_of(plan.frames, n_frames, context);
    {
        scratch work(AUDIO8_MAX_NODES);
        if (!work.ok()) return false;
        const synthesis_graph built = build_synthesis(work.ctx, model, out.block_span);
        mark_output(work.graph, built.pcm);
        ::tts_cpp::detail::fit_graph_price price;
        if (!fit_price_graph(model.backend, work.graph, 2 * AUDIO8_MAX_NODES, price)) {
            return false;
        }
        out.device_bytes = sat_add(out.device_bytes, price.device_bytes);
        out.host_bytes   = sat_add(out.host_bytes, price.host_bytes);
    }
    return true;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
