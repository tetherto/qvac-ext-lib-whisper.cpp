#include "parakeet/engine.h"
#include "parakeet/streaming.h"
#include "parakeet_ctc.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

namespace {

bool throws_not_implemented(const std::function<void()> & action) {
    try {
        action();
    } catch (const std::runtime_error & error) {
        return std::string(error.what()).find("not implemented") !=
            std::string::npos;
    }
    return false;
}

int check_loaded_model(const std::string & path) {
    parakeet::ParakeetCtcModel model;
    const int rc = parakeet::load_from_gguf(
        path, model, 0, 0, false);
    if (rc != 0) {
        std::fprintf(stderr, "load_from_gguf failed: %d\n", rc);
        return 1;
    }
    if (model.model_type != parakeet::ParakeetModelType::NEMOTRON ||
        std::string(parakeet::model_type_name(model.model_type)) !=
            "nemotron") {
        std::fprintf(stderr, "Nemotron family dispatch failed\n");
        return 2;
    }
    if (model.rnnt.predict_embed || model.tdt.predict_embed ||
        !model.nemotron.rnnt.predict_embed ||
        !model.nemotron.prompt.proj_0_w ||
        !model.nemotron.prompt.proj_2_w) {
        std::fprintf(stderr, "Nemotron tensor ownership failed\n");
        return 3;
    }
    if (parakeet::resolve_nemotron_prompt_id(model, "en-US") != 0 ||
        parakeet::resolve_nemotron_prompt_id(model, "auto") != 101 ||
        parakeet::resolve_nemotron_prompt_id(model, "") != 101) {
        std::fprintf(stderr, "Nemotron locale resolution failed\n");
        return 4;
    }

    bool rejected_locale = false;
    try {
        (void) parakeet::resolve_nemotron_prompt_id(
            model, "invalid-locale");
    } catch (const std::runtime_error &) {
        rejected_locale = true;
    }
    if (!rejected_locale) {
        std::fprintf(stderr, "invalid Nemotron locale was accepted\n");
        return 5;
    }

    parakeet::ParakeetCtcModel malformed_metadata = model;
    malformed_metadata.nemotron_cfg.prompt_width = 127;
    bool rejected_metadata = false;
    try {
        parakeet::validate_nemotron_model(malformed_metadata);
    } catch (const std::runtime_error &) {
        rejected_metadata = true;
    }
    if (!rejected_metadata) {
        std::fprintf(stderr, "malformed Nemotron metadata was accepted\n");
        return 6;
    }

    parakeet::ParakeetCtcModel malformed_tensor = model;
    malformed_tensor.nemotron.prompt.proj_0_w = nullptr;
    bool rejected_tensor = false;
    try {
        parakeet::validate_nemotron_model(malformed_tensor);
    } catch (const std::runtime_error &) {
        rejected_tensor = true;
    }
    if (!rejected_tensor) {
        std::fprintf(stderr, "malformed Nemotron model was accepted\n");
        return 7;
    }
    return 0;
}

int check_engine_guard(const std::string & path) {
    parakeet::EngineOptions options;
    options.model_gguf_path = path;
    options.prewarm = false;
    parakeet::Engine engine(options);
    if (engine.model_type() != "nemotron" ||
        !engine.is_transcription_model()) {
        std::fprintf(stderr, "Engine did not expose Nemotron family\n");
        return 8;
    }

    const float sample = 0.0f;
    parakeet::StreamingOptions streaming_options;
    const parakeet::StreamingCallback callback =
        [](const parakeet::StreamingSegment &) {};
    if (!throws_not_implemented([&]() {
        (void) engine.transcribe_samples_stream(
            &sample, 1, 16000, streaming_options, callback);
    })) {
        std::fprintf(stderr, "Nemotron streaming inference guard did not fire\n");
        return 9;
    }
    if (!throws_not_implemented([&]() {
        (void) engine.stream_start(streaming_options, callback);
    })) {
        std::fprintf(stderr, "Nemotron stream session guard did not fire\n");
        return 10;
    }
    return 0;
}

}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <nemotron.gguf>\n", argv[0]);
        return 2;
    }

    if (const int rc = check_loaded_model(argv[1]); rc != 0) {
        return rc;
    }
    if (const int rc = check_engine_guard(argv[1]); rc != 0) {
        return rc;
    }

    std::fprintf(stderr, "Nemotron loader tests passed\n");
    return 0;
}
