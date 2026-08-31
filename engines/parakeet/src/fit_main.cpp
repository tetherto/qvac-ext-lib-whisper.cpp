// parakeet-fit-params CLI: project whether a Parakeet GGUF fits the device
// memory available right now, without reading weight data. Thin front-end
// over parakeet::fit_params (include/parakeet/fit.h); lives in the library so
// hosts can link `parakeet_fit_cli_main` directly (same shape as
// parakeet_cli_main). Exit code == fit status: 0 fits, 1 does not fit,
// 2 error.

#include "parakeet/cli.h"
#include "parakeet/fit.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void print_usage(const char * argv0) {
    std::printf(
        "usage: %s --model MODEL.gguf [options]\n"
        "\n"
        "Projects the model's memory needs against the free device memory,\n"
        "reading only GGUF metadata (no weights are loaded, nothing runs).\n"
        "Exit code: 0 = fits, 1 = does not fit, 2 = error.\n"
        "\n"
        "options:\n"
        "  --model PATH            Parakeet GGUF to project (required)\n"
        "  --audio-seconds S       workload: longest single transcribe input\n"
        "                          (default 300; device memory saturates at the\n"
        "                          long-form window for transcription models,\n"
        "                          but keeps growing for Sortformer diarization)\n"
        "  --n-gpu-layers N        request the GPU backend when > 0 (default 0 = CPU),\n"
        "                          with the same runtime fallbacks a real load applies\n"
        "  --threads N             CPU thread count used for backend resolution\n"
        "  --margin-mib MIB        free-memory headroom to require (default 256)\n"
        "  --window-frames N       EngineOptions::long_form_window_frames (default 0 = auto)\n"
        "  --context-frames N      EngineOptions::long_form_context_frames (default 0 = auto)\n"
        "  --backends-dir DIR      directory scanned for dynamically-loaded ggml backends\n"
        "  --json                  emit the projection as JSON on stdout\n"
        "  --verbose               loader diagnostics on stderr\n"
        "  --help                  this text\n",
        argv0);
}

void print_json(const parakeet::FitResult & r, uint64_t margin_bytes) {
    auto b = [](bool v) { return v ? "true" : "false"; };
    std::printf("{\n");
    std::printf("  \"status\": %d,\n", (int) r.status);
    std::printf("  \"statusName\": \"%s\",\n", parakeet::fit_status_name(r.status));
    std::printf("  \"fits\": %s,\n", b(r.fits));
    std::printf("  \"reason\": \"%s\",\n", r.reason.c_str());
    std::printf("  \"modelType\": \"%s\",\n", r.model_type.c_str());
    std::printf("  \"modelVariant\": \"%s\",\n", r.model_variant.c_str());
    std::printf("  \"deviceName\": \"%s\",\n", r.device_name.c_str());
    std::printf("  \"deviceIsCpu\": %s,\n", b(r.device_is_cpu));
    std::printf("  \"deviceFreeBytes\": %" PRIu64 ",\n", r.device_free_bytes);
    std::printf("  \"deviceTotalBytes\": %" PRIu64 ",\n", r.device_total_bytes);
    std::printf("  \"weightsBytes\": %" PRIu64 ",\n", r.device.weights_bytes);
    std::printf("  \"encoderComputeBytes\": %" PRIu64 ",\n", r.device.encoder_compute_bytes);
    std::printf("  \"decoderStateBytes\": %" PRIu64 ",\n", r.device.decoder_state_bytes);
    std::printf("  \"decoderComputeBytes\": %" PRIu64 ",\n", r.device.decoder_compute_bytes);
    std::printf("  \"deviceProjectedBytes\": %" PRIu64 ",\n", r.device.total_bytes);
    std::printf("  \"hostBytes\": %" PRIu64 ",\n", r.host_bytes);
    std::printf("  \"marginBytes\": %" PRIu64 "\n", margin_bytes);
    std::printf("}\n");
}

}  // namespace

extern "C" int parakeet_fit_cli_main(int argc, char ** argv) {
    parakeet::FitOptions opts;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--model" && i + 1 < argc) {
            opts.model_gguf_path = argv[++i];
        } else if (a == "--audio-seconds" && i + 1 < argc) {
            opts.audio_seconds = (float) std::atof(argv[++i]);
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            opts.n_gpu_layers = std::atoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            opts.n_threads = std::atoi(argv[++i]);
        } else if (a == "--margin-mib" && i + 1 < argc) {
            opts.margin_bytes = (uint64_t) std::strtoull(argv[++i], nullptr, 10) * 1024 * 1024;
        } else if (a == "--window-frames" && i + 1 < argc) {
            opts.long_form_window_frames = std::atoi(argv[++i]);
        } else if (a == "--context-frames" && i + 1 < argc) {
            opts.long_form_context_frames = std::atoi(argv[++i]);
        } else if (a == "--backends-dir" && i + 1 < argc) {
            opts.backends_dir = argv[++i];
        } else if (a == "--json") {
            json = true;
        } else if (a == "--verbose" || a == "-v") {
            opts.verbose = true;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n\n", a.c_str());
            print_usage(argv[0]);
            return (int) parakeet::FitStatus::Error;
        }
    }

    if (opts.model_gguf_path.empty()) {
        std::fprintf(stderr, "--model is required\n\n");
        print_usage(argv[0]);
        return (int) parakeet::FitStatus::Error;
    }

    const parakeet::FitResult r = parakeet::fit_params(opts);

    if (json) {
        print_json(r, opts.margin_bytes);
    } else if (r.status == parakeet::FitStatus::Error) {
        std::fprintf(stderr, "parakeet-fit-params: error: %s\n", r.reason.c_str());
    } else {
        std::printf("%s", r.report.c_str());
    }

    return (int) r.status;
}
