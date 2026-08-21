#include "minimax/logic.h"
#include "minimax/backend.h"
#include "minimax/bpe.h"
#include "minimax/mm3-flow-runtime.h"
#include "minimax/mm3-replay-io.h"
#include "minimax/mm3-window-orchestrator.h"
#include "minimax/progress.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

#define CHECK(condition)                                                                      \
    do {                                                                                      \
        ++checks;                                                                             \
        if (!(condition)) {                                                                   \
            ++failures;                                                                       \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition);          \
        }                                                                                     \
    } while (0)

template <typename Function>
bool throws_invalid_argument(Function function) {
    try {
        function();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

template <typename Function>
bool throws_runtime_error(Function function) {
    try {
        function();
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// MSVC has no POSIX setenv/unsetenv; _putenv_s(key, "") removes the variable,
// which matches backend_configure_device treating an empty value as unset.
void set_env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) {
        setenv(key, value, 1);
    } else {
        unsetenv(key);
    }
#endif
}

bool close(float left, float right, float tolerance = 1e-6f) {
    return std::fabs(left - right) <= tolerance;
}

tts_cpp::minimax::detail::ModelCompatibility valid_compatibility() {
    using namespace tts_cpp::minimax::detail;
    ModelCompatibility model;
    model.lm_embedding = 4096;
    model.lm_codebooks = 8;
    model.lm_acoustic_vocab = 1024;
    model.frame_rate = 25;
    model.max_audio_frames = 9000;
    model.max_prompt_tokens = 5000;
    model.depth_embedding = 4096;
    model.depth_codebooks = 8;
    model.depth_acoustic_vocab = 1024;
    model.condition_layers = 8;
    model.condition_hidden = 4096;
    model.condition_out = 2048;
    model.condition_rate = {24000, 960, 44100, 512};
    model.dit_condition = 2048;
    model.dit_channels = 128;
    model.window_frames = 200;
    model.hop_frames = 100;
    model.window_latents = 689;
    model.hop_latents = 344;
    model.vocoder_latent_channels = 128;
    model.vocoder_sampling_rate = 44100;
    model.vocoder_channels = 2;
    model.vocoder_upsample = 512;
    model.components = {"depth", "cond", "dit", "vocoder"};
    return model;
}

void touch(const std::filesystem::path & path) {
    std::ofstream stream(path);
    stream << "test";
}

void test_frame_validation() {
    using namespace tts_cpp::minimax::detail;
    CHECK(kDefaultFrameRate == 25);
    CHECK(kDefaultMaxFrames == 9000);
    CHECK(frames_from_duration(12.0, 25, 9000) == 300);
    CHECK(frames_from_duration(1000.0, 25, 9000) == 9000);
    CHECK(validate_frames(1, 9000) == 1);
    CHECK(validate_frames(9001, 9000) == 9000);
    CHECK(throws_invalid_argument([] { frames_from_duration(0.0, 25, 9000); }));
    CHECK(throws_invalid_argument([] { frames_from_duration(std::numeric_limits<double>::infinity(), 25, 9000); }));
    CHECK(throws_invalid_argument([] { validate_frames(0, 9000); }));
}

void test_prompt() {
    using tts_cpp::minimax::detail::build_prompt;
    const std::string prompt = build_prompt("## **Bright** pop", "[Verse] ignored\nHello ^ world");
    CHECK(prompt == "<|im_start|><|caption_start|>Bright pop<|caption_end|><|lyrics_start|>"
                    "[start]\n[verse]\nHello\nworld<|lyrics_end|><|im_end|><|audio_start|>");
    CHECK(build_prompt("Instrumental piano", "") ==
          "<|im_start|><|caption_start|>Instrumental piano<|caption_end|><|lyrics_start|>"
          "[start]\n[instrumental]<|lyrics_end|><|im_end|><|audio_start|>");
    CHECK(throws_invalid_argument([] { build_prompt(" \n\t", "words"); }));
}

void test_unconditional_mask() {
    using tts_cpp::minimax::detail::mask_unconditional;
    const std::vector<int32_t> conditional = {10, 11, 12, 13, 14, 15};
    CHECK(mask_unconditional(conditional, 99) == std::vector<int32_t>({10, 99, 99, 99, 14, 15}));
    CHECK(mask_unconditional({1, 2, 3}, 99) == std::vector<int32_t>({1, 2, 3}));
}

void test_noise() {
    using tts_cpp::minimax::detail::fill_noise;
    std::vector<float> first;
    std::vector<float> second;
    std::vector<float> other;
    fill_noise(42, 0, first, 8);
    fill_noise(42, 0, second, 8);
    fill_noise(42, 1, other, 8);
    CHECK(first == second);
    CHECK(first != other);
    CHECK(close(first[0], -0.19663458f));
    CHECK(close(first[1], -0.25129682f));
    CHECK(throws_invalid_argument([&] { fill_noise(42, -1, first, 8); }));
}

void test_flow_schedule() {
    using tts_cpp::minimax::detail::flow_schedule;
    CHECK(tts_cpp::minimax::detail::kDefaultFlowSteps == 30);
    CHECK(close(tts_cpp::minimax::detail::kDefaultCfgScale, 1.7f));
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    flow_schedule(4, sigmas, timesteps);
    CHECK(sigmas.size() == 5);
    CHECK(timesteps.size() == 4);
    CHECK(close(timesteps[0], 0.0f));
    CHECK(close(timesteps[1], 0.25f));
    CHECK(close(timesteps[2], 0.5f));
    CHECK(close(timesteps[3], 0.75f));
    CHECK(close(sigmas[4], 1.0f));
    CHECK(throws_invalid_argument([&] { flow_schedule(0, sigmas, timesteps); }));
}

void test_production_dit_readback_preserves_velocity() {
    const std::vector<float> raw = {1.0f, -2.0f, 0.5f};
    size_t requested_bytes = 0;
    const MM3DitReadback readback =
        [&raw, &requested_bytes](float * output, size_t bytes, std::string *) {
            requested_bytes = bytes;
            std::copy(raw.begin(), raw.end(), output);
            return true;
        };
    std::vector<float> output(raw.size());
    std::string error;
    CHECK(mm3_read_dit_output(output.data(), output.size(), readback, &error));
    CHECK(error.empty());
    CHECK(requested_bytes == raw.size() * sizeof(float));
    CHECK(output == raw);
}

void test_production_cfg_euler_step() {
    std::vector<float> latents = {1.0f, -1.0f};
    const std::vector<float> conditional = {2.0f, 4.0f};
    const std::vector<float> unconditional = {0.5f, -2.0f};
    const std::vector<float> sigmas = {0.0f, 0.25f};
    std::string error;
    CHECK(mm3_integrate_flow_step(latents, conditional, unconditional, sigmas, 0,
                                  1.5f, &error));
    CHECK(error.empty());
    CHECK(latents == std::vector<float>({1.6875f, 0.75f}));
}

tts_cpp::minimax::detail::SynthesisContract valid_synthesis_contract() {
    tts_cpp::minimax::detail::SynthesisContract contract;
    contract.scheduler = "FlowMatchEulerDiscrete";
    contract.invert_sigmas = true;
    contract.shift = 1.0f;
    contract.train_timesteps = 1;
    contract.rope_type = "neox";
    contract.glu_order = "value_gate";
    contract.timestep_token_prepended = true;
    contract.pre_post_conv_residual = true;
    contract.attn_bias = false;
    return contract;
}

void test_malformed_synthesis_metadata() {
    using namespace tts_cpp::minimax::detail;
    CHECK(vocoder_upsample_error({8, 8, 2, 4}, 512).empty());
    CHECK(!vocoder_upsample_error({}, 512).empty());
    CHECK(!vocoder_upsample_error({8, 0, 2, 4}, 512).empty());
    CHECK(!vocoder_upsample_error({8, -1, 2, 4}, 512).empty());
    CHECK(!vocoder_upsample_error({8, 8, 2, 4}, 256).empty());
    CHECK(!vocoder_upsample_error({std::numeric_limits<int32_t>::max(), 3}, 1).empty());

    SynthesisContract contract = valid_synthesis_contract();
    CHECK(validate_synthesis_contract(contract).empty());
    contract.scheduler = "Other";
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.invert_sigmas = false;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.shift = std::numeric_limits<float>::quiet_NaN();
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.shift = 1.0001f;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.train_timesteps = 1000;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.rope_type = "interleaved";
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.glu_order = "gate_value";
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.timestep_token_prepended = false;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.pre_post_conv_residual = false;
    CHECK(!validate_synthesis_contract(contract).empty());
    contract = valid_synthesis_contract();
    contract.attn_bias = true;
    CHECK(!validate_synthesis_contract(contract).empty());
}

void test_vocoder_output_shape() {
    using tts_cpp::minimax::detail::vocoder_output_shape_error;
    CHECK(vocoder_output_shape_error(8, 1, 1, 1, 8, 8 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(7, 1, 1, 1, 8, 8 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(8, 2, 1, 1, 8, 8 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(8, 1, 1, 1, 7, 8 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(8, 1, 1, 1, 8, 7 * sizeof(float), 8).empty());
    CHECK(!vocoder_output_shape_error(0, 1, 1, 1, 0, 0, 0).empty());
}

void test_copy_ranges() {
    using namespace tts_cpp::minimax::detail;
    CHECK(copy_range_fits(10, 8, 2, 1, 4));
    CHECK(!copy_range_fits(10, 8, -1, 1, 4));
    CHECK(!copy_range_fits(10, 8, 8, 1, 3));
    CHECK(!copy_range_fits(10, 8, 2, 7, 2));
    CHECK(stereo_copy_ranges_fit(20, 16, 10, 2, 8, 1, 4));
    CHECK(!stereo_copy_ranges_fit(19, 16, 10, 2, 8, 1, 4));
    CHECK(!stereo_copy_ranges_fit(20, 16, 10, 7, 8, 1, 4));
    CHECK(!stereo_copy_ranges_fit(20, 16, 10, 2, 8, 6, 4));
}

void test_condition_length() {
    using namespace tts_cpp::minimax::detail;
    ConditionRate rate;
    CHECK(condition_latent_length(rate, 200) == 689);
    CHECK(condition_latent_length(rate, 100) == 344);
    CHECK(condition_latent_length(rate, 1) == 3);
    CHECK(throws_invalid_argument([&] { condition_latent_length(rate, 0); }));
}

void test_window_arithmetic() {
    using namespace tts_cpp::minimax::detail;
    CHECK(window_starts(200, kWindowFrames, kHopFrames) == std::vector<int64_t>({0}));
    CHECK(window_starts(300, kWindowFrames, kHopFrames) == std::vector<int64_t>({0, 100}));
    CHECK(window_starts(301, kWindowFrames, kHopFrames) == std::vector<int64_t>({0, 100, 200}));
    const CropSpan first = crop_span(689, 0, 2, 512);
    const CropSpan second = crop_span(689, 1, 2, 512);
    CHECK(first.left == 0);
    CHECK(first.length == 431 * 512);
    CHECK(second.left == 86 * 512);
    CHECK(second.length == 603 * 512);
    CHECK(stitched_sample_count({689, 689}, 512) == 529408);
    CHECK(kCarryLatents == kCropLeftLatents + kCropRightLatents);
    CHECK(kCarryLatents == 2 * kBlendLatents);
}

void test_overlap_blend_and_pin() {
    using namespace tts_cpp::minimax::detail;
    std::vector<float> noise = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    std::vector<float> previous = {20.0f, 40.0f, 60.0f, 80.0f};
    std::vector<float> latents = noise;
    blend_latent_overlap(latents.data(), noise.data(), previous.data(), 2, 3, 2, 2,
                         0.5f);
    CHECK(close(latents[0], 11.000001f));
    CHECK(close(latents[1], 22.000002f));
    CHECK(close(latents[3], 34.000004f));
    CHECK(close(latents[4], 45.000005f));
    pin_latent_overlap(latents.data(), previous.data(), 2, 3, 2, 2);
    CHECK(latents == std::vector<float>({20.0f, 40.0f, 6.0f, 60.0f, 80.0f, 12.0f}));
}

void test_carry_layout() {
    using namespace tts_cpp::minimax::detail;
    std::vector<float> latents(2 * 689);
    std::vector<float> condition(3 * 689);
    for (size_t index = 0; index < latents.size(); ++index) {
        latents[index] = static_cast<float>(index);
    }
    for (size_t index = 0; index < condition.size(); ++index) {
        condition[index] = static_cast<float>(index);
    }
    const CarryRange range = carry_range(689, 344, 172);
    CHECK(range.start == 345);
    CHECK(range.end == 517);
    std::vector<float> carry_latents;
    std::vector<float> carry_condition;
    CHECK(copy_carry_layout(latents, condition, 2, 3, 689, range, carry_latents,
                            carry_condition));
    CHECK(carry_latents.size() == 344);
    CHECK(carry_latents[0] == 345.0f);
    CHECK(carry_latents[171] == 516.0f);
    CHECK(carry_latents[172] == 1034.0f);
    CHECK(carry_condition.size() == 516);
    CHECK(carry_condition.front() == 1035.0f);
    CHECK(carry_condition.back() == 1550.0f);
}

void test_planar_stitch() {
    using tts_cpp::minimax::detail::copy_planar_window;
    const std::vector<float> source = {0.0f, 1.0f, 2.0f, 3.0f,
                                       10.0f, 11.0f, 12.0f, 13.0f};
    std::vector<float> destination(12, -1.0f);
    CHECK(copy_planar_window(source, 2, 4, 1, 6, 2, 2, destination));
    CHECK(destination == std::vector<float>({-1.0f, -1.0f, 1.0f, 2.0f, -1.0f, -1.0f,
                                             -1.0f, -1.0f, 11.0f, 12.0f, -1.0f, -1.0f}));
}

void emit_flow_steps(int64_t steps,
                     const std::function<void(int64_t, int64_t)> & on_step) {
    for (int64_t step = 1; step <= steps; ++step) {
        on_step(step, steps);
    }
}

int count_stage(const std::vector<std::string> & stages, const std::string & stage) {
    return static_cast<int>(std::count(stages.begin(), stages.end(), stage));
}

void fill_fake_latents(int64_t window, int64_t channels, int64_t latent_length,
                       int64_t overlap, std::vector<float> & latents) {
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t index = overlap; index < latent_length; ++index) {
            latents[static_cast<size_t>(channel * latent_length + index)] =
                static_cast<float>(window * 10000 + channel * 1000 + index);
        }
    }
}

constexpr int64_t kTwoWindowChannels = 2;
constexpr int64_t kTwoWindowConditionDimension = 2;
constexpr int64_t kTwoWindowUpsample = 2;
constexpr int64_t kTwoWindowFlowSteps = 2;
constexpr int64_t kTwoWindowFrames = 201;

struct TwoWindowCapture {
    std::vector<std::string> stages;
    std::vector<float> first_condition;
    std::vector<float> blended;
    std::vector<float> pinned;
};

MM3WindowDimensions make_two_window_dimensions() {
    MM3WindowDimensions dimensions;
    dimensions.latent_channels = kTwoWindowChannels;
    dimensions.audio_channels = kTwoWindowChannels;
    dimensions.condition_dimension = kTwoWindowConditionDimension;
    dimensions.upsample = kTwoWindowUpsample;
    dimensions.window_frames = 200;
    dimensions.hop_frames = 100;
    dimensions.carry_span = 344;
    dimensions.overlap = 172;
    dimensions.crop_left = 86;
    dimensions.crop_right = 258;
    dimensions.flow_steps = kTwoWindowFlowSteps;
    return dimensions;
}

bool capture_two_window_progress(TwoWindowCapture & capture,
                                 const std::string & stage) {
    capture.stages.push_back(stage);
    return true;
}

void fill_two_window_condition(int64_t window, std::vector<float> & condition) {
    for (size_t index = 0; index < condition.size(); ++index) {
        condition[index] =
            static_cast<float>(window * 10000 + static_cast<int64_t>(index));
    }
}

bool inject_two_window_condition(TwoWindowCapture & capture, int64_t window,
                                 int64_t frames, std::vector<float> & condition,
                                 int64_t & latent_length) {
    latent_length =
        tts_cpp::minimax::detail::condition_latent_length({}, frames);
    condition.resize(static_cast<size_t>(
        latent_length * kTwoWindowConditionDimension));
    fill_two_window_condition(window, condition);
    if (window == 0) {
        capture.first_condition = condition;
    }
    return true;
}

bool inject_two_window_noise(int64_t window, int64_t count,
                             std::vector<float> & noise) {
    noise.assign(static_cast<size_t>(count), window == 0 ? 0.25f : 0.5f);
    return window == 1;
}

void capture_two_window_overlap(TwoWindowCapture & capture, int64_t latent_length,
                                const std::vector<float> & noise,
                                const std::vector<float> & carry_latents,
                                int64_t carry_length,
                                const std::vector<float> & condition,
                                std::vector<float> & latents) {
    CHECK(carry_length == 172);
    CHECK(condition[0] ==
          capture.first_condition[345 * kTwoWindowConditionDimension]);
    CHECK(condition[343] ==
          capture.first_condition[345 * kTwoWindowConditionDimension + 343]);
    tts_cpp::minimax::detail::blend_latent_overlap(
        latents.data(), noise.data(), carry_latents.data(), kTwoWindowChannels,
        latent_length, carry_length, carry_length, 0.5f);
    capture.blended = latents;
    tts_cpp::minimax::detail::pin_latent_overlap(
        latents.data(), carry_latents.data(), kTwoWindowChannels, latent_length,
        carry_length, carry_length);
    capture.pinned = latents;
}

bool inject_two_window_flow(
    TwoWindowCapture & capture, int64_t window, int64_t latent_length,
    const std::vector<float> & noise, const std::vector<float> & carry_latents,
    int64_t carry_length, const std::vector<float> & condition,
    std::vector<float> & latents,
    const std::function<void(int64_t, int64_t)> & on_step) {
    latents = noise;
    if (window == 1) {
        capture_two_window_overlap(capture, latent_length, noise, carry_latents,
                                   carry_length, condition, latents);
    }
    fill_fake_latents(window, kTwoWindowChannels, latent_length, carry_length,
                      latents);
    emit_flow_steps(kTwoWindowFlowSteps, on_step);
    return true;
}

bool inject_two_window_vocoder(int64_t window, int64_t latent_length,
                               std::vector<float> & waveform) {
    const int64_t samples = latent_length * kTwoWindowUpsample;
    waveform.resize(static_cast<size_t>(samples * kTwoWindowChannels));
    std::fill(waveform.begin(), waveform.begin() + samples,
              window == 0 ? 0.1f : 0.3f);
    std::fill(waveform.begin() + samples, waveform.end(),
              window == 0 ? 0.2f : 0.4f);
    return true;
}

MM3WindowOperations make_two_window_operations(TwoWindowCapture & capture) {
    MM3WindowOperations operations;
    operations.progress =
        [&capture](const std::string & stage, int64_t, int64_t, int64_t, int64_t,
                   std::string *) {
            return capture_two_window_progress(capture, stage);
        };
    operations.condition =
        [&capture](int64_t window, int64_t, int64_t, int64_t frames,
                   std::vector<float> & condition, int64_t & latent_length,
                   std::string *) {
            return inject_two_window_condition(capture, window, frames, condition,
                                               latent_length);
        };
    operations.noise =
        [](int64_t window, int64_t count, std::vector<float> & noise) {
            return inject_two_window_noise(window, count, noise);
        };
    operations.flow =
        [&capture](
            int64_t window, int64_t, int64_t latent_length,
            const std::vector<float> & noise,
            const std::vector<float> & carry_latents, int64_t carry_length,
            const std::vector<float> & condition, std::vector<float> & latents,
            const std::function<void(int64_t, int64_t)> & on_step,
            std::string *) {
            return inject_two_window_flow(
                capture, window, latent_length, noise, carry_latents, carry_length,
                condition, latents, on_step);
        };
    operations.vocoder =
        [](int64_t window, int64_t, const std::vector<float> &, int64_t latent_length,
           std::vector<float> & waveform, std::string *) {
            return inject_two_window_vocoder(window, latent_length, waveform);
        };
    return operations;
}

void assert_two_window_expected_state(const MM3WindowOrchestration & result,
                                      const std::string & error) {
    CHECK(error.empty());
    CHECK(result.starts == std::vector<int64_t>({0, 100}));
    CHECK(result.frame_lengths == std::vector<int64_t>({200, 101}));
    CHECK(result.latent_lengths == std::vector<int64_t>({689, 347}));
    CHECK(result.overlaps == std::vector<int64_t>({0, 172}));
    CHECK(result.carry_starts[0] == 345);
    CHECK(result.carry_ends[0] == 517);
    CHECK(result.vocoder_calls == 2);
    CHECK(result.samples_per_channel == 1384);
    CHECK(result.audio.size() == 2768);
    CHECK(result.forced_noise == std::vector<int64_t>({0, 1}));
}

void assert_two_window_crops(const MM3WindowDimensions & dimensions) {
    const auto first_crop = mm3_window_crop_span(dimensions, 689, 0, 2);
    const auto second_crop = mm3_window_crop_span(dimensions, 347, 1, 2);
    CHECK(first_crop.left == 0);
    CHECK(first_crop.length == 862);
    CHECK(second_crop.left == 172);
    CHECK(second_crop.length == 522);
}

void assert_two_window_seams(const MM3WindowOrchestration & result,
                             const TwoWindowCapture & capture) {
    CHECK(close(result.audio[861], 0.1f));
    CHECK(close(result.audio[862], 0.3f));
    CHECK(close(result.audio[1384 + 861], 0.2f));
    CHECK(close(result.audio[1384 + 862], 0.4f));
    CHECK(!capture.blended.empty());
    CHECK(!capture.pinned.empty());
    CHECK(!close(capture.blended[0], capture.pinned[0]));
    CHECK(capture.pinned[0] == result.latents[0][345]);
}

void assert_two_window_progress(const TwoWindowCapture & capture) {
    CHECK(count_stage(capture.stages, "cond") == 2);
    CHECK(count_stage(capture.stages, "flow") == 6);
    CHECK(count_stage(capture.stages, "vocode") == 2);
    CHECK(count_stage(capture.stages, "stitch") == 1);
    CHECK(count_stage(capture.stages, "done") == 1);
}

void test_two_window_orchestration() {
    const MM3WindowDimensions dimensions = make_two_window_dimensions();
    TwoWindowCapture capture;
    const MM3WindowOperations operations = make_two_window_operations(capture);
    MM3WindowOrchestration result;
    std::string error;
    CHECK(mm3_orchestrate_windows(kTwoWindowFrames, dimensions, operations, &result,
                                  &error));
    assert_two_window_expected_state(result, error);
    assert_two_window_crops(dimensions);
    assert_two_window_seams(result, capture);
    assert_two_window_progress(capture);
}

void test_nonfinite_vocoder_orchestration(float value) {
    const MM3WindowDimensions dimensions = make_two_window_dimensions();
    TwoWindowCapture capture;
    MM3WindowOperations operations = make_two_window_operations(capture);
    operations.vocoder =
        [value](int64_t, int64_t, const std::vector<float> &, int64_t latent_length,
                std::vector<float> & waveform, std::string *) {
            const int64_t samples = latent_length * kTwoWindowUpsample;
            waveform.assign(static_cast<size_t>(samples * kTwoWindowChannels), 0.25f);
            waveform[4] = value;
            return true;
        };
    MM3WindowOrchestration result;
    std::string error;
    CHECK(!mm3_orchestrate_windows(1, dimensions, operations, &result, &error));
    CHECK(error == "non-finite audio sample at index 4");
    CHECK(result.audio.size() == 12);
    CHECK(!std::isfinite(result.audio[4]));
}

void test_nonfinite_vocoder_orchestration() {
    test_nonfinite_vocoder_orchestration(std::numeric_limits<float>::quiet_NaN());
    test_nonfinite_vocoder_orchestration(std::numeric_limits<float>::infinity());
    test_nonfinite_vocoder_orchestration(-std::numeric_limits<float>::infinity());
}

void test_finite_vocoder_orchestration_clipping() {
    const MM3WindowDimensions dimensions = make_two_window_dimensions();
    TwoWindowCapture capture;
    MM3WindowOperations operations = make_two_window_operations(capture);
    operations.vocoder =
        [](int64_t, int64_t, const std::vector<float> &, int64_t latent_length,
           std::vector<float> & waveform, std::string *) {
            const int64_t samples = latent_length * kTwoWindowUpsample;
            waveform.assign(static_cast<size_t>(samples * kTwoWindowChannels), 0.0f);
            waveform[0] = 2.0f;
            waveform[1] = -2.0f;
            waveform[2] = 0.5f;
            return true;
        };
    MM3WindowOrchestration result;
    std::string error;
    CHECK(mm3_orchestrate_windows(1, dimensions, operations, &result, &error));
    CHECK(error.empty());
    CHECK(result.audio[0] == 1.0f);
    CHECK(result.audio[1] == -1.0f);
    CHECK(result.audio[2] == 0.5f);
    CHECK(close(result.metrics.peak, 1.0f));
    CHECK(std::fabs(result.metrics.rms - std::sqrt(0.1875)) <= 1e-12);
}

void test_ar_candidate_helpers() {
    using namespace tts_cpp::minimax::detail;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> logits = {
        0.0f, 9.0f, 0.0f, nan, inf, -inf,
        0.0f, 4.0f, 0.0f, 1.0f, 2.0f, 3.0f,
    };
    ArCandidates candidates;
    int64_t nonfinite = 0;
    CHECK(collect_ar_candidates(logits.data(), 6, 1, 3, 3, candidates, nonfinite));
    CHECK(candidates.conditional.size() == 4);
    CHECK(candidates.conditional[0] == 9.0f);
    CHECK(std::isinf(candidates.conditional[1]) && candidates.conditional[1] < 0.0f);
    CHECK(std::isinf(candidates.conditional[2]) && candidates.conditional[2] < 0.0f);
    CHECK(std::isinf(candidates.conditional[3]) && candidates.conditional[3] < 0.0f);
    CHECK(candidates.unconditional == std::vector<float>({4.0f, 1.0f, 2.0f, 3.0f}));
    CHECK(nonfinite == 2);

    candidates.conditional = {4.0f, 3.0f, 3.0f, 1.0f};
    candidates.unconditional = {0.0f, 0.0f, 0.0f, 0.0f};
    guide_ar_candidates(candidates, 2.0f);
    apply_conditional_top_k(candidates, 2);
    CHECK(candidates.guided[0] == 8.0f);
    CHECK(candidates.guided[1] == 6.0f);
    CHECK(candidates.guided[2] == 6.0f);
    CHECK(std::isinf(candidates.guided[3]) && candidates.guided[3] < 0.0f);
}

void test_ar_acoustic_rows_and_frame_cap() {
    using namespace tts_cpp::minimax::detail;
    const int32_t codes[] = {2, 3, 4};
    std::vector<int32_t> rows;
    build_acoustic_rows(codes, 3, 10, rows);
    CHECK(rows == std::vector<int32_t>({2, 13, 24}));
    std::string error;
    CHECK(resolve_ar_frame_cap(20, 12, true, 6, error) == 5);
    CHECK(error.empty());
    CHECK(resolve_ar_frame_cap(20, 12, false, 0, error) == 12);
    CHECK(resolve_ar_frame_cap(20, 12, true, 1, error) == 0);
    CHECK(!error.empty());
}

void test_sampling_edges() {
    using tts_cpp::minimax::detail::sample_top_k;
    std::mt19937_64 random(42);
    CHECK(sample_top_k(nullptr, 0, 50, random) == 0);
    const float logits[] = {-5.0f, 3.0f, 2.0f};
    CHECK(sample_top_k(logits, 3, 1, random) == 1);
    const float unusual[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
    CHECK(sample_top_k(unusual, 3, 1, random) == 1);
    std::mt19937_64 first(7);
    std::mt19937_64 second(7);
    CHECK(sample_top_k(logits, 3, 3, first) == sample_top_k(logits, 3, 3, second));
}

void test_model_compatibility() {
    using namespace tts_cpp::minimax::detail;
    ModelCompatibility model = valid_compatibility();
    CHECK(validate_model_compatibility(model).empty());
    model.condition_out = 1024;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.dit_channels = 64;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.depth_codebooks = 7;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.components.push_back("vocoder");
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.condition_rate.output_sampling_rate = 32000;
    CHECK(!validate_model_compatibility(model).empty());
}

void test_unicode_categories() {
    CHECK(is_digit(0x0661));
    CHECK(is_letter(0x4E2D));
    CHECK(is_letter(0x00E9));
    CHECK(!is_letter(0x1F600));
    CHECK(!is_digit(0x1F600));
    CHECK(is_letter(0x10400));
    CHECK(is_whitespace(0x2003));
    CHECK(!is_whitespace(0x200B));
    CHECK(gpt2_pre_tokenize(u8"\u00E9\u4E2D\U00010400") ==
          std::vector<std::string>({u8"\u00E9\u4E2D\U00010400"}));
    CHECK(gpt2_pre_tokenize(u8"\u0661\u0662\u0663\u0664") ==
          std::vector<std::string>({u8"\u0661", u8"\u0662", u8"\u0663", u8"\u0664"}));
    CHECK(gpt2_pre_tokenize(u8"\U0001F600\u00E9") ==
          std::vector<std::string>({u8"\U0001F600\u00E9"}));
    CHECK(gpt2_pre_tokenize(u8"\u2003\u4E2D") ==
          std::vector<std::string>({u8"\u2003\u4E2D"}));
}

void test_pre_tokenization_branches() {
    CHECK(gpt2_pre_tokenize(u8"I'LL we\u2019RE they've dog's can't I'm he'd") ==
          std::vector<std::string>({"I", "'LL", " we", u8"\u2019RE", " they", "'ve",
                                    " dog", "'s", " can", "'t", " I", "'m", " he", "'d"}));
    CHECK(gpt2_pre_tokenize("'llama") == std::vector<std::string>({"'llama"}));
    CHECK(gpt2_pre_tokenize(u8"Hello\u00E9\u4E2D\U00010400") ==
          std::vector<std::string>({u8"Hello\u00E9\u4E2D\U00010400"}));
    CHECK(gpt2_pre_tokenize(u8".world") == std::vector<std::string>({u8".world"}));
    CHECK(gpt2_pre_tokenize(u8"\u0667") == std::vector<std::string>({u8"\u0667"}));
    CHECK(gpt2_pre_tokenize("!?\r\n") == std::vector<std::string>({"!?\r\n"}));
    CHECK(gpt2_pre_tokenize("  hello") == std::vector<std::string>({" ", " hello"}));
    CHECK(gpt2_pre_tokenize(" \t!") == std::vector<std::string>({" ", "\t!"}));
    CHECK(gpt2_pre_tokenize("  7") == std::vector<std::string>({" ", " ", "7"}));
    CHECK(gpt2_pre_tokenize(" \t\n") == std::vector<std::string>({" \t\n"}));
    CHECK(gpt2_pre_tokenize(u8"\u2003\u2003\u4E2D") ==
          std::vector<std::string>({u8"\u2003", u8"\u2003\u4E2D"}));
}

void test_malformed_utf8() {
    const std::string truncated_four_byte("\xF0", 1);
    const std::string truncated_sequence("\xF0\x9F", 2);
    const std::string invalid_prefix("\x80" "A", 2);
    CHECK(gpt2_pre_tokenize(truncated_four_byte) ==
          std::vector<std::string>({truncated_four_byte}));
    CHECK(gpt2_pre_tokenize(truncated_sequence) ==
          std::vector<std::string>({std::string("\xF0", 1), std::string("\x9F", 1)}));
    CHECK(gpt2_pre_tokenize(invalid_prefix) ==
          std::vector<std::string>({invalid_prefix}));
    int advance = 0;
    CHECK(utf8_codepoint(truncated_sequence.data(),
                         static_cast<int>(truncated_sequence.size()), &advance) == 0xF0);
    CHECK(advance == 1);
}

void test_model_pair_resolution() {
    namespace fs = std::filesystem;
    using tts_cpp::minimax::detail::ModelPair;
    using tts_cpp::minimax::detail::resolve_model_pair;
    const fs::path root = fs::path("/tmp/tether") /
                          ("minimax-model-pair-" + std::to_string(std::random_device{}()));
    fs::create_directories(root / "mm3");
    touch(root / "mm3-lm-f16.gguf");
    touch(root / "mm3-synth-f16.gguf");
    touch(root / "MM3-LM-Q8_0.GGUF");
    touch(root / "mm3-SYNTH-q8_0.gguf");
    ModelPair pair = resolve_model_pair(root.string(), "", "");
    CHECK(pair.quant == "q8_0");
    CHECK(fs::path(pair.lm).filename() == "MM3-LM-Q8_0.GGUF");
    touch(root / "mm3" / "mm3-lm-q8_0.gguf");
    bool ambiguous = false;
    try {
        resolve_model_pair(root.string(), "", "");
    } catch (const std::runtime_error &) {
        ambiguous = true;
    }
    CHECK(ambiguous);
    fs::remove_all(root);
}

void test_backend_configuration() {
    using tts_cpp::minimax::detail::backend_configuration_matches;
    CHECK(backend_configuration_matches(0, 4, "first", 8, "second"));
    CHECK(backend_configuration_matches(1, 4, "backends/.", 4, "backends"));
    CHECK(!backend_configuration_matches(1, 4, "backends", 8, "backends"));
    CHECK(!backend_configuration_matches(1, 4, "first", 4, "second"));
}

void test_device_configuration() {
    CHECK(throws_runtime_error([] { backend_configure_device("fast"); }));

    set_env("MM3_DEVICE", "auto");
    backend_configure_device("cpu");
    CHECK(g_backend_device == "cpu");

    backend_configure_device("");
    CHECK(g_backend_device == "auto");

    set_env("MM3_DEVICE", "vulkan");
    CHECK(throws_runtime_error([] { backend_configure_device(""); }));

    set_env("MM3_DEVICE", nullptr);
    backend_configure_device("");
    CHECK(g_backend_device == "cpu");
}

void test_device_backend_init() {
    ggml_backend_load_all();
    ggml_backend_t probe = tts_cpp::acestep::backend_gpu_init();
    const bool gpu_available = probe != nullptr;
    if (probe) {
        ggml_backend_free(probe);
    }

    backend_configure_device("cpu");
    BackendPair cpu_pair = backend_init("test");
    CHECK(cpu_pair.cpu_backend != nullptr);
    CHECK(!cpu_pair.has_gpu);
    CHECK(cpu_pair.backend == cpu_pair.cpu_backend);
    CHECK(throws_runtime_error([] { backend_configure_device("auto"); }));
    backend_release(cpu_pair.backend, cpu_pair.cpu_backend);

    backend_configure_device("auto");
    BackendPair auto_pair = backend_init("test");
    CHECK(auto_pair.cpu_backend != nullptr);
    CHECK(auto_pair.has_gpu == gpu_available);
    backend_release(auto_pair.backend, auto_pair.cpu_backend);

    backend_configure_device("gpu");
    if (gpu_available) {
        BackendPair gpu_pair = backend_init("test");
        CHECK(gpu_pair.has_gpu);
        CHECK(gpu_pair.backend != gpu_pair.cpu_backend);
        backend_release(gpu_pair.backend, gpu_pair.cpu_backend);
    } else {
        CHECK(throws_runtime_error([] { backend_init("test"); }));
    }
    backend_configure_device("cpu");
}

void test_replay_io() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "mm3-replay-io-test";
    fs::remove_all(dir);
    std::string error;
    CHECK(mm3_replay_prepare_output_dir(dir.string(), &error));
    CHECK(mm3_replay_prepare_output_dir(dir.string(), &error));

    const std::vector<float> payload = {1.0f, -2.5f, 0.25f};
    const std::string raw_path = (dir / "data.f32").string();
    CHECK(mm3_replay_write_raw(raw_path, payload.data(), payload.size()));
    std::vector<float> loaded;
    CHECK(mm3_replay_read_raw(raw_path, loaded));
    CHECK(loaded == payload);

    const std::vector<float> planar = {0.5f, -0.5f, 1.5f, -1.5f};
    const std::string wav_path = (dir / "audio.wav").string();
    CHECK(mm3_replay_write_wav(wav_path, planar, 2, 44100));
    CHECK(fs::file_size(wav_path) == 44 + 2 * 2 * sizeof(int16_t));

    const std::string missing = (dir / "missing-subdir" / "data.f32").string();
    CHECK(!mm3_replay_write_raw(missing, payload.data(), payload.size()));
    CHECK(!mm3_replay_write_wav(missing, planar, 2, 44100));

    std::string not_a_dir_error;
    CHECK(!mm3_replay_prepare_output_dir(raw_path, &not_a_dir_error));
    CHECK(!not_a_dir_error.empty());

    std::vector<float> absent;
    CHECK(!mm3_replay_read_raw((dir / "absent.bin").string(), absent));

    const std::string empty_path = (dir / "empty.f32").string();
    CHECK(mm3_replay_write_raw(empty_path, payload.data(), 0));
    std::vector<float> from_empty;
    CHECK(!mm3_replay_read_raw(empty_path, from_empty));

    const std::string truncated_path = (dir / "truncated.f32").string();
    const char truncated_bytes[5] = {1, 2, 3, 4, 5};
    CHECK(mm3_replay_write_raw(truncated_path, truncated_bytes, sizeof(truncated_bytes)));
    std::vector<float> from_truncated;
    CHECK(!mm3_replay_read_raw(truncated_path, from_truncated));

    CHECK(mm3_replay_mode_is_supported("full"));
    CHECK(mm3_replay_mode_is_supported("replay"));
    CHECK(mm3_replay_mode_is_supported("condcheck"));
    CHECK(!mm3_replay_mode_is_supported("ful"));
    CHECK(!mm3_replay_mode_is_supported(""));
    fs::remove_all(dir);
}

void test_engine_instance_limit() {
    using tts_cpp::minimax::detail::engine_instance_available;
    CHECK(engine_instance_available(0));
    CHECK(!engine_instance_available(1));
}

void test_cancellation() {
    using tts_cpp::minimax::detail::cancellation_requested;
    CHECK(!cancellation_requested({}));
    CHECK(cancellation_requested([] { return true; }));
    CHECK(!cancellation_requested([] { return false; }));
}

void test_progress_cancellation() {
    bool cancelled = false;
    std::string error;
    const MM3ProgressCb progress = [&cancelled](const MM3GenProgress &) {
        cancelled = true;
    };
    CHECK(!mm3_emit_progress(progress, {"stitch", -1, 1, 0, 1},
                             [&cancelled] { return cancelled; }, &error));
    CHECK(error == MM3_ERR_CANCELLED);
}

}

int main() {
    test_frame_validation();
    test_prompt();
    test_unconditional_mask();
    test_noise();
    test_flow_schedule();
    test_production_dit_readback_preserves_velocity();
    test_production_cfg_euler_step();
    test_malformed_synthesis_metadata();
    test_vocoder_output_shape();
    test_copy_ranges();
    test_condition_length();
    test_window_arithmetic();
    test_overlap_blend_and_pin();
    test_carry_layout();
    test_planar_stitch();
    test_two_window_orchestration();
    test_nonfinite_vocoder_orchestration();
    test_finite_vocoder_orchestration_clipping();
    test_ar_candidate_helpers();
    test_ar_acoustic_rows_and_frame_cap();
    test_sampling_edges();
    test_model_compatibility();
    test_unicode_categories();
    test_pre_tokenization_branches();
    test_malformed_utf8();
    test_model_pair_resolution();
    test_backend_configuration();
    test_device_configuration();
    test_device_backend_init();
    test_replay_io();
    test_engine_instance_limit();
    test_cancellation();
    test_progress_cancellation();
    std::fprintf(stderr, "[test-minimax-units] %d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
