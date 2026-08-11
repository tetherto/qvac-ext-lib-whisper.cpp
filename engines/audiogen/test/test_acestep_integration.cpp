#include "audiogen-cpp/acestep/engine.h"

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

int failures = 0;

void check(bool condition, const char * expression) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", expression);
}

#define CHECK(condition) check((condition), #condition)

struct StageLog {
    std::vector<std::string> names;

    bool record(const std::string & stage, int, int) {
        names.push_back(stage);
        return true;
    }

    bool contains(const char * stage) const {
        return std::find(names.begin(), names.end(), stage) != names.end();
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
