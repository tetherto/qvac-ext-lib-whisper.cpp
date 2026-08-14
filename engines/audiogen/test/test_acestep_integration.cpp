#include "audiogen-cpp/acestep/engine.h"

#include "audio_edit.h"
#include "cover_noise.h"
#include "dit_ggml.h"
#include "generate_task.h"
#include "philox.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int TEST_SAMPLE_RATE = 48000;
constexpr int TEST_CHANNELS = 2;
constexpr int TEST_SECONDS = 2;
constexpr int TEST_STEPS = 2;
constexpr float TEST_SHIFT = 3.0f;
constexpr float TEST_NOISE_STRENGTH = 0.5f;
constexpr long long TEST_SEED = 22886;
constexpr float TEST_FREQUENCY = 220.0f;
constexpr float PI = 3.14159265358979323846f;
constexpr int TEST_LATENT_CHANNELS = tts_cpp::acestep::AUDIO_EDIT_LATENT_CHANNELS;
constexpr int TEST_CONTEXT_BANKS = 2;
constexpr int TEST_CONTEXT_CHANNELS = TEST_LATENT_CHANNELS * TEST_CONTEXT_BANKS;
constexpr float TEST_CONTEXT_ACTIVE_VALUE = 1.0f;
constexpr float TEST_REPAINT_START_SECONDS = 0.5f;
constexpr float TEST_REPAINT_END_SECONDS = 1.5f;
constexpr float TEST_PRESERVED_LEFT_SECONDS = 0.4f;
constexpr float TEST_PRESERVED_RIGHT_SECONDS = 1.6f;
constexpr float TEST_FLOW_MAX_RATIO = 0.5f;
constexpr char TEST_CONTEXT_DUMP[] = "03_context.bin";
constexpr char TEST_SOURCE_DUMP[] = "00_source_latent.bin";
constexpr char TEST_REPAINT_CAPTION[] = "Replace the middle with bright synth guitar";
constexpr char TEST_FLOW_SOURCE_CAPTION[] = "A plain sine tone";
constexpr char TEST_FLOW_TARGET_CAPTION[] = "A warm evolving synth pad";
constexpr char TEST_UNKNOWN_LANGUAGE[] = "unknown";
constexpr char TEST_LM_STAGE[] = "lm";
constexpr char TEST_TURBO_ERROR[] = "turbo DiT only";
constexpr char TEST_FLOW_SKIP_FORMAT[] =
    "[test-acestep-integration] flow-edit skipped: supplied model is not turbo\n";

int failures = 0;

void check(bool condition, const char * expression) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", expression);
}

#define CHECK(condition) check((condition), #condition)

struct StageLog {
    std::vector<std::string> names;
    struct Event {
      std::string name;
      int step;
      int total;
    };
    std::vector<Event> events;

    bool record(const std::string &stage, int step, int total) {
      names.push_back(stage);
      events.push_back({stage, step, total});
      return true;
    }

    bool contains(const char * stage) const {
        return std::find(names.begin(), names.end(), stage) != names.end();
    }

    bool contains_detailed_progress(const char *stage) const {
      return std::any_of(events.begin(), events.end(), [&](const Event &event) {
        return event.name == stage && event.total > 1 && event.step > 0;
      });
    }
};

struct TensorDump {
    std::vector<float> values;
    int rows = 0;
    int columns = 0;
};

std::vector<float> make_test_pcm() {
    const int frames = TEST_SAMPLE_RATE * TEST_SECONDS;
    std::vector<float> pcm((size_t) frames * TEST_CHANNELS);
    for (int frame = 0; frame < frames; ++frame) {
        const float sample = 0.1f * std::sin(2.0f * PI * TEST_FREQUENCY *
                                            (float) frame / TEST_SAMPLE_RATE);
        pcm[(size_t) frame * TEST_CHANNELS] = sample;
        pcm[(size_t) frame * TEST_CHANNELS + 1] = sample;
    }
    return pcm;
}

fs::path make_dump_directory() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path path = fs::temp_directory_path() / ("audiogen-cover-integration-" + std::to_string(stamp));
    fs::create_directories(path);
    return path;
}

TensorDump read_tensor_dump(const fs::path & path) {
    TensorDump dump;
    FILE * file = std::fopen(path.string().c_str(), "rb");
    if (!file) return dump;

    int32_t header[3] = { 0, 0, 0 };
    if (std::fread(header, sizeof(int32_t), 3, file) != 3) {
        std::fclose(file);
        return dump;
    }

    dump.rows = header[1];
    dump.columns = header[2];
    dump.values.resize((size_t) dump.rows * dump.columns);
    const size_t read = std::fread(dump.values.data(), sizeof(float), dump.values.size(), file);
    std::fclose(file);
    if (read != dump.values.size()) dump = {};
    return dump;
}

bool vectors_match(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) return false;
    for (size_t index = 0; index < left.size(); ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

tts_cpp::acestep::EngineOptions make_engine_options(const char * models_dir, const fs::path & dump_dir) {
    tts_cpp::acestep::EngineOptions options;
    options.models_dir = models_dir;
    options.dump_stages_dir = dump_dir.string();
    options.n_threads = 4;
    options.n_gpu_layers = std::getenv("AUDIOGEN_TEST_GPU") ? 99 : 0;
    return options;
}

tts_cpp::acestep::GenerateParams make_generate_params() {
    tts_cpp::acestep::GenerateParams params;
    params.caption = "Instrumental cover integration test";
    params.lyrics = "[Instrumental]";
    params.source_audio = make_test_pcm();
    params.reference_audio = params.source_audio;
    params.task_type = tts_cpp::acestep::TASK_COVER_NOFSQ;
    params.audio_cover_strength = 1.0f;
    params.cover_noise_strength = TEST_NOISE_STRENGTH;
    params.inference_steps = TEST_STEPS;
    params.shift = TEST_SHIFT;
    params.seed = TEST_SEED;
    params.lm_phase1 = false;
    return params;
}

void verify_cover_noise(const fs::path & dump_dir) {
    using namespace tts_cpp::acestep;

    const TensorDump source = read_tensor_dump(dump_dir / "00_source_latent.bin");
    const TensorDump noise = read_tensor_dump(dump_dir / "07_noise.bin");
    CHECK(!source.values.empty());
    CHECK(!noise.values.empty());
    if (source.values.empty() || noise.values.empty()) return;

    std::vector<float> expected(noise.values.size());
    philox_randn(TEST_SEED, expected.data(), (int) expected.size(), true);
    std::vector<float> schedule;
    dit_build_schedule(TEST_SHIFT, TEST_STEPS, schedule);
    apply_cover_noise(expected, source.values, noise.rows, source.rows, noise.columns,
                      TEST_NOISE_STRENGTH, schedule);
    CHECK(vectors_match(expected, noise.values));
}

void verify_stage_dumps(const fs::path & dump_dir) {
    CHECK(fs::exists(dump_dir / "00_source_latent.bin"));
    CHECK(fs::exists(dump_dir / "00_reference_latent.bin"));
    CHECK(!fs::exists(dump_dir / "01_lm_codes.bin"));
    CHECK(!fs::exists(dump_dir / "02_detok_latent.bin"));
    CHECK(fs::exists(dump_dir / "07_noise.bin"));
    verify_cover_noise(dump_dir);
}

void inspect_flow_context_channels(const TensorDump & context, const TensorDump & source,
                                   int frame, bool & differs_from_source, bool & varies_over_time) {
    for (int channel = 0; channel < TEST_LATENT_CHANNELS; ++channel) {
        const size_t context_index = (size_t) frame * TEST_CONTEXT_CHANNELS + channel;
        if (context.values[context_index] != context.values[(size_t) channel]) {
            varies_over_time = true;
        }
        if (frame < source.rows &&
            source.values[(size_t) frame * source.columns + channel] != context.values[context_index]) {
            differs_from_source = true;
        }
        CHECK(context.values[context_index + TEST_LATENT_CHANNELS] == TEST_CONTEXT_ACTIVE_VALUE);
    }
}

void inspect_flow_context_frames(const TensorDump & context, const TensorDump & source,
                                 bool & differs_from_source, bool & varies_over_time) {
    for (int frame = 0; frame < context.rows; ++frame) {
        inspect_flow_context_channels(context, source, frame, differs_from_source, varies_over_time);
    }
}

void verify_flow_silence_context(const fs::path & dump_dir) {
    const TensorDump context = read_tensor_dump(dump_dir / TEST_CONTEXT_DUMP);
    const TensorDump source = read_tensor_dump(dump_dir / TEST_SOURCE_DUMP);
    CHECK(!context.values.empty());
    CHECK(context.columns == TEST_CONTEXT_CHANNELS);
    CHECK(context.rows > 1);
    CHECK(!source.values.empty());
    CHECK(source.columns == TEST_LATENT_CHANNELS);
    if (context.values.empty() || context.columns != TEST_CONTEXT_CHANNELS || context.rows <= 1 ||
        source.values.empty() || source.columns != TEST_LATENT_CHANNELS) {
        return;
    }
    bool differs_from_source = false;
    bool varies_over_time = false;
    inspect_flow_context_frames(context, source, differs_from_source, varies_over_time);
    CHECK(differs_from_source);
    CHECK(varies_over_time);
}

bool repaint_rows_differ(const float * left, const float * right) {
    for (int channel = 0; channel < TEST_LATENT_CHANNELS; ++channel) {
        if (left[channel] != right[channel]) return true;
    }
    return false;
}

void inspect_repaint_context_frames(const TensorDump & context, const float *& first_repaint_frame,
                                    bool & found_repaint_frame, bool & varies_over_time) {
    for (int frame = 0; frame < context.rows; ++frame) {
        const float * row = context.values.data() + (size_t) frame * context.columns;
        if (row[TEST_LATENT_CHANNELS] != TEST_CONTEXT_ACTIVE_VALUE) continue;
        found_repaint_frame = true;
        if (!first_repaint_frame) {
            first_repaint_frame = row;
            continue;
        }
        if (repaint_rows_differ(row, first_repaint_frame)) varies_over_time = true;
    }
}

void verify_repaint_silence_context(const fs::path & dump_dir) {
    const TensorDump context = read_tensor_dump(dump_dir / TEST_CONTEXT_DUMP);
    CHECK(!context.values.empty());
    CHECK(context.columns == TEST_CONTEXT_CHANNELS);
    if (context.values.empty() || context.columns != TEST_CONTEXT_CHANNELS) return;
    const float * first_repaint_frame = nullptr;
    bool found_repaint_frame = false;
    bool varies_over_time = false;
    inspect_repaint_context_frames(context, first_repaint_frame, found_repaint_frame, varies_over_time);
    CHECK(found_repaint_frame);
    CHECK(varies_over_time);
}

struct EditFixture {
    tts_cpp::acestep::GenerateParams params;
    tts_cpp::acestep::RepaintParams repaint;
};

tts_cpp::acestep::GenerateResult generate_with_stage_log(tts_cpp::acestep::Engine & engine,
                                                         tts_cpp::acestep::GenerateParams & params,
                                                         StageLog & stages) {
    return engine.generate(params, [&](const std::string & stage, int step, int total) {
        return stages.record(stage, step, total);
    });
}

EditFixture make_edit_fixture() {
    using namespace tts_cpp::acestep;
    EditFixture fixture;
    fixture.params.caption = TEST_REPAINT_CAPTION;
    fixture.params.lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    fixture.params.source_audio = make_test_pcm();
    fixture.params.reference_audio = fixture.params.source_audio;
    fixture.params.inference_steps = TEST_STEPS;
    fixture.params.shift = TEST_SHIFT;
    fixture.params.seed = TEST_SEED;
    fixture.params.lm_phase1 = false;
    fixture.repaint.start_seconds = TEST_REPAINT_START_SECONDS;
    fixture.repaint.end_seconds = TEST_REPAINT_END_SECONDS;
    fixture.repaint.mode = RepaintMode::Balanced;
    fixture.repaint.strength = REPAINT_DEFAULT_STRENGTH;
    fixture.params.edit_plan.emplace_back(fixture.repaint);
    return fixture;
}

void check_preserved_pcm_range(const std::vector<float> & actual, const std::vector<float> & expected,
                               int first_frame, int last_frame) {
    for (int frame = first_frame; frame < last_frame; ++frame) {
        for (int channel = 0; channel < TEST_CHANNELS; ++channel) {
            const size_t index = (size_t) frame * TEST_CHANNELS + channel;
            CHECK(actual[index] == expected[index]);
        }
    }
}

void verify_repaint_preservation(const tts_cpp::acestep::GenerateResult & result,
                                 const tts_cpp::acestep::GenerateParams & params) {
    const int preserved_left_frames = (int) (TEST_PRESERVED_LEFT_SECONDS * TEST_SAMPLE_RATE);
    const int preserved_right_start = (int) (TEST_PRESERVED_RIGHT_SECONDS * TEST_SAMPLE_RATE);
    const int total_frames = TEST_SECONDS * TEST_SAMPLE_RATE;
    check_preserved_pcm_range(result.pcm, params.source_audio, 0, preserved_left_frames);
    check_preserved_pcm_range(result.pcm, params.source_audio, preserved_right_start, total_frames);
}

void run_repaint_scenario(tts_cpp::acestep::Engine & engine, const fs::path & dump_dir,
                          EditFixture & fixture) {
    using namespace tts_cpp::acestep;
    StageLog stages;
    const GenerateResult result = generate_with_stage_log(engine, fixture.params, stages);
    CHECK(!result.pcm.empty());
    CHECK(stages.contains(AUDIO_EDIT_REPAINT_STAGE));
    CHECK(!stages.contains(TEST_LM_STAGE));
    CHECK(stages.contains_detailed_progress("vae"));
    CHECK(result.metadata.vocal_language == TEST_UNKNOWN_LANGUAGE);
    verify_repaint_silence_context(dump_dir);
    verify_repaint_preservation(result, fixture.params);
}

tts_cpp::acestep::GenerateParams make_flow_params(
    const tts_cpp::acestep::GenerateParams & repaint_params) {
    using namespace tts_cpp::acestep;
    GenerateParams params = repaint_params;
    params.edit_plan.clear();
    FlowEditParams flow;
    flow.source_caption = TEST_FLOW_SOURCE_CAPTION;
    flow.source_lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    flow.target_caption = TEST_FLOW_TARGET_CAPTION;
    flow.target_lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    flow.n_min = AUDIO_EDIT_MIN_RATIO;
    flow.n_max = TEST_FLOW_MAX_RATIO;
    flow.n_avg = FLOW_EDIT_DEFAULT_AVERAGES;
    params.edit_plan.emplace_back(flow);
    return params;
}

void verify_composed_stage_order(const StageLog & stages) {
    using namespace tts_cpp::acestep;
    const auto flow_position =
        std::find(stages.names.begin(), stages.names.end(), AUDIO_EDIT_FLOW_STAGE);
    const auto repaint_position =
        std::find(stages.names.begin(), stages.names.end(), AUDIO_EDIT_REPAINT_STAGE);
    CHECK(flow_position != stages.names.end());
    CHECK(repaint_position != stages.names.end());
    CHECK(flow_position < repaint_position);
}

void run_composed_scenario(tts_cpp::acestep::Engine & engine,
                           tts_cpp::acestep::GenerateParams & params,
                           const tts_cpp::acestep::RepaintParams & repaint) {
    params.edit_plan.emplace_back(repaint);
    StageLog stages;
    const tts_cpp::acestep::GenerateResult result =
        generate_with_stage_log(engine, params, stages);
    CHECK(!result.pcm.empty());
    verify_composed_stage_order(stages);
}

void run_flow_scenario(tts_cpp::acestep::Engine & engine, const fs::path & dump_dir,
                       const EditFixture & fixture) {
    using namespace tts_cpp::acestep;
    GenerateParams params = make_flow_params(fixture.params);
    StageLog stages;
    try {
        const GenerateResult result = generate_with_stage_log(engine, params, stages);
        CHECK(!result.pcm.empty());
        CHECK(stages.contains(AUDIO_EDIT_FLOW_STAGE));
        CHECK(!stages.contains(TEST_LM_STAGE));
        verify_flow_silence_context(dump_dir);
        run_composed_scenario(engine, params, fixture.repaint);
    } catch (const std::invalid_argument & error) {
        if (std::string(error.what()).find(TEST_TURBO_ERROR) == std::string::npos) throw;
        std::fprintf(stderr, TEST_FLOW_SKIP_FORMAT);
    }
}

void run_edit_vae_cancel_scenario(tts_cpp::acestep::Engine &engine,
                                  const EditFixture &fixture) {
  bool cancelled_during_vae = false;
  const tts_cpp::acestep::GenerateResult result = engine.generate(
      fixture.params, [&](const std::string &stage, int step, int total) {
        if (stage == "vae" && total > 1 && step > 0) {
          cancelled_during_vae = true;
          return false;
        }
        return true;
      });
  CHECK(cancelled_during_vae);
  CHECK(result.pcm.empty());
}

void run_edit_scenarios(tts_cpp::acestep::Engine & engine, const fs::path & dump_dir) {
    EditFixture fixture = make_edit_fixture();
    run_repaint_scenario(engine, dump_dir, fixture);
    run_flow_scenario(engine, dump_dir, fixture);
    run_edit_vae_cancel_scenario(engine, fixture);
}

int run_integration(const char * models_dir) {
    using namespace tts_cpp::acestep;

    const fs::path dump_dir = make_dump_directory();
    try {
        std::unique_ptr<Engine> engine = Engine::create(make_engine_options(models_dir, dump_dir));
        GenerateParams params = make_generate_params();
        StageLog stages;
        const GenerateResult result =
            engine->generate(params, [&](const std::string & stage, int step, int total) {
                return stages.record(stage, step, total);
            });

        CHECK(!result.pcm.empty());
        CHECK(result.sample_rate == TEST_SAMPLE_RATE);
        CHECK(result.channels == TEST_CHANNELS);
        CHECK(result.metadata.n_codes == 0);
        CHECK(stages.contains("source"));
        CHECK(stages.contains("reference"));
        CHECK(!stages.contains("lm"));
        verify_stage_dumps(dump_dir);
        run_edit_scenarios(*engine, dump_dir);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL integration exception: %s\n", error.what());
        ++failures;
    }
    fs::remove_all(dump_dir);
    return failures == 0 ? 0 : 1;
}

}

int main() {
    const char * models_dir = std::getenv("AUDIOGEN_TEST_MODELS_DIR");
    if (!models_dir || !*models_dir) {
        std::fprintf(stderr, "[test-acestep-integration] skipped: AUDIOGEN_TEST_MODELS_DIR is unset\n");
        return 77;
    }
    return run_integration(models_dir);
}
