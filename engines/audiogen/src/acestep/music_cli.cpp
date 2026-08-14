// music-cli: end-to-end ACE-Step text-to-music harness.
//
// Drives tts_cpp::acestep::Engine (text prompt -> LM codes -> FSQ detok ->
// text/cond encoders -> DiT flow-matching -> VAE) and writes a 48 kHz stereo
// WAV. This is the first fully-native music generation path (no acestep.cpp
// binaries) — the same C++ that the @qvac/audiogen-ggml addon links.
//
// Usage:
//   music-cli --models <dir>            [--out song.wav] [--dur 8] [--seed 42]
//   music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf ...
//   optional: --caption "..." --lyrics "..." --steps 8 --shift 3.0
//             --bpm 128 --key "C major" --tsig 4/4 --lang en
//             --gpu --threads N --dump-stages <existing dir>

#include "audiogen-cpp/acestep/engine.h"
#include "wav_reader.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static const char * arg_val(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return nullptr;
}

static bool arg_flag(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], key)) return true;
    return false;
}

// --- tiny JSON field readers (flat object; good enough for a request json) ---
static std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s((size_t) n, '\0');
    size_t rd = fread(&s[0], 1, (size_t) n, f);
    fclose(f);
    s.resize(rd);
    return s;
}

// Return the raw value token after "key": . Handles "string" or bareword/number.
static bool json_field(const std::string & j, const char * key, std::string & out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t      p      = j.find(needle);
    if (p == std::string::npos) return false;
    p = j.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < j.size() && (j[p] == ' ' || j[p] == '\t' || j[p] == '\n' || j[p] == '\r')) p++;
    if (p >= j.size()) return false;
    if (j[p] == '"') {
        size_t e = ++p;
        std::string v;
        while (e < j.size() && j[e] != '"') {
            if (j[e] == '\\' && e + 1 < j.size()) {
                switch (j[e + 1]) {
                    case '"':  v += '"';  break;
                    case '\\': v += '\\'; break;
                    case '/':  v += '/';  break;
                    case 'b':  v += '\b'; break;
                    case 'f':  v += '\f'; break;
                    case 'n':  v += '\n'; break;
                    case 'r':  v += '\r'; break;
                    case 't':  v += '\t'; break;
                    default:   v += j[e + 1]; break;
                }
                e += 2;
                continue;
            }
            v += j[e++];
        }
        out = v;
    } else {
        size_t e = p;
        while (e < j.size() && j[e] != ',' && j[e] != '}' && j[e] != '\n') e++;
        out = j.substr(p, e - p);
        while (!out.empty() && (out.back() == ' ' || out.back() == '\r' || out.back() == '\t')) out.pop_back();
    }
    return true;
}

static std::vector<std::string> json_array_objects(const std::string & json, const char * key) {
    std::vector<std::string> objects;
    const std::string needle = std::string("\"") + key + "\"";
    size_t cursor = json.find(needle);
    if (cursor == std::string::npos) return objects;
    cursor = json.find('[', cursor + needle.size());
    if (cursor == std::string::npos) return objects;

    bool in_string = false;
    bool escaped = false;
    int object_depth = 0;
    size_t object_start = std::string::npos;
    for (++cursor; cursor < json.size(); ++cursor) {
        const char ch = json[cursor];
        if (in_string) {
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') in_string = false;
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            if (object_depth++ == 0) object_start = cursor;
        } else if (ch == '}') {
            if (--object_depth == 0 && object_start != std::string::npos) {
                objects.push_back(json.substr(object_start, cursor - object_start + 1));
                object_start = std::string::npos;
            }
        } else if (ch == ']' && object_depth == 0) {
            break;
        }
    }
    return objects;
}

static constexpr int kRequiredAudioSampleRate = 48000;
static constexpr const char * kEditPlanOption = "--edit-plan";
static constexpr const char * kNormalizeOption = "--normalize";
static constexpr const char * kRepaintStartOption = "--repaint-start";
static constexpr const char * kRepaintEndOption = "--repaint-end";
static constexpr const char * kRepaintModeOption = "--repaint-mode";
static constexpr const char * kRepaintStrengthOption = "--repaint-strength";
static constexpr const char * kFlowSourceCaptionOption = "--flow-source-caption";
static constexpr const char * kFlowSourceLyricsOption = "--flow-source-lyrics";
static constexpr const char * kFlowMinimumOption = "--flow-n-min";
static constexpr const char * kFlowMaximumOption = "--flow-n-max";
static constexpr const char * kFlowAverageOption = "--flow-n-avg";
static constexpr const char * kSourceAudioOption = "--src-audio";
static constexpr const char * kTextToMusicTask = "text2music";
static constexpr const char * kOperationsField = "operations";
static constexpr const char * kTypeField = "type";
static constexpr const char * kRepaintType = "repaint";
static constexpr const char * kFlowEditType = "flow-edit";
static constexpr const char * kStartField = "start";
static constexpr const char * kEndField = "end";
static constexpr const char * kStrengthField = "strength";
static constexpr const char * kCaptionField = "caption";
static constexpr const char * kLyricsField = "lyrics";
static constexpr const char * kModeField = "mode";
static constexpr const char * kSourceCaptionField = "source_caption";
static constexpr const char * kSourceLyricsField = "source_lyrics";
static constexpr const char * kTargetCaptionField = "target_caption";
static constexpr const char * kTargetLyricsField = "target_lyrics";
static constexpr const char * kMinimumField = "n_min";
static constexpr const char * kMaximumField = "n_max";
static constexpr const char * kAverageField = "n_avg";
static constexpr const char * kConservativeMode = "conservative";
static constexpr const char * kBalancedMode = "balanced";
static constexpr const char * kAggressiveMode = "aggressive";

static bool parse_repaint_mode(const std::string & value,
                               tts_cpp::acestep::RepaintMode & mode) {
    using tts_cpp::acestep::RepaintMode;
    if (value == kConservativeMode) mode = RepaintMode::Conservative;
    else if (value == kBalancedMode) mode = RepaintMode::Balanced;
    else if (value == kAggressiveMode) mode = RepaintMode::Aggressive;
    else return false;
    return true;
}

static bool parse_repaint_operation(
        const std::string & operation,
        std::vector<tts_cpp::acestep::AudioEditParams> & plan,
        std::string & error) {
    using namespace tts_cpp::acestep;
    RepaintParams repaint;
    std::string value;
    if (json_field(operation, kStartField, value)) repaint.start_seconds = (float) atof(value.c_str());
    if (json_field(operation, kEndField, value)) repaint.end_seconds = (float) atof(value.c_str());
    if (json_field(operation, kStrengthField, value)) repaint.strength = (float) atof(value.c_str());
    if (json_field(operation, kCaptionField, value)) repaint.caption = value;
    if (json_field(operation, kLyricsField, value)) repaint.lyrics = value;
    if (json_field(operation, kModeField, value) &&
        !parse_repaint_mode(value, repaint.mode)) {
        error = "repaint mode must be conservative|balanced|aggressive";
        return false;
    }
    plan.emplace_back(std::move(repaint));
    return true;
}

static void parse_flow_edit_operation(
        const std::string & operation,
        std::vector<tts_cpp::acestep::AudioEditParams> & plan) {
    using namespace tts_cpp::acestep;
    FlowEditParams flow;
    std::string value;
    if (json_field(operation, kSourceCaptionField, value)) flow.source_caption = value;
    if (json_field(operation, kSourceLyricsField, value)) flow.source_lyrics = value;
    if (json_field(operation, kTargetCaptionField, value)) flow.target_caption = value;
    if (json_field(operation, kTargetLyricsField, value)) flow.target_lyrics = value;
    if (json_field(operation, kMinimumField, value)) flow.n_min = (float) atof(value.c_str());
    if (json_field(operation, kMaximumField, value)) flow.n_max = (float) atof(value.c_str());
    if (json_field(operation, kAverageField, value)) flow.n_avg = atoi(value.c_str());
    plan.emplace_back(std::move(flow));
}

static bool parse_edit_operation(
        const std::string & operation,
        std::vector<tts_cpp::acestep::AudioEditParams> & plan,
        std::string & error) {
    std::string type;
    if (!json_field(operation, kTypeField, type)) {
        error = "edit plan operation is missing type";
        return false;
    }
    if (type == kRepaintType) return parse_repaint_operation(operation, plan, error);
    if (type == kFlowEditType) {
        parse_flow_edit_operation(operation, plan);
        return true;
    }
    error = "unsupported edit plan operation type '" + type + "'";
    return false;
}

static bool parse_edit_operations(
        const std::vector<std::string> & operations,
        std::vector<tts_cpp::acestep::AudioEditParams> & plan,
        std::string & error) {
    if (operations.empty()) {
        error = "edit plan requires a non-empty operations array";
        return false;
    }
    for (const std::string & operation : operations) {
        if (!parse_edit_operation(operation, plan, error)) return false;
    }
    return true;
}

static bool load_edit_plan(const char * path,
                           std::vector<tts_cpp::acestep::AudioEditParams> & plan,
                           std::string & error) {
    const std::string json = read_file(path);
    if (json.empty()) {
        error = "cannot read edit plan";
        return false;
    }
    return parse_edit_operations(json_array_objects(json, kOperationsField), plan, error);
}

static bool has_standalone_repaint_flags(int argc, char ** argv) {
    return arg_val(argc, argv, kRepaintStartOption) ||
           arg_val(argc, argv, kRepaintEndOption) ||
           arg_val(argc, argv, kRepaintModeOption) ||
           arg_val(argc, argv, kRepaintStrengthOption);
}

static bool has_standalone_flow_flags(int argc, char ** argv) {
    return arg_val(argc, argv, kFlowSourceCaptionOption) != nullptr;
}

static bool parse_standalone_repaint(
        int argc, char ** argv,
        std::vector<tts_cpp::acestep::AudioEditParams> & plan) {
    using namespace tts_cpp::acestep;
    if (!has_standalone_repaint_flags(argc, argv)) return true;
    RepaintParams repaint;
    if (const char * value = arg_val(argc, argv, kRepaintStartOption))
        repaint.start_seconds = (float) atof(value);
    if (const char * value = arg_val(argc, argv, kRepaintEndOption))
        repaint.end_seconds = (float) atof(value);
    if (const char * value = arg_val(argc, argv, kRepaintStrengthOption))
        repaint.strength = (float) atof(value);
    if (const char * value = arg_val(argc, argv, kRepaintModeOption)) {
        if (!parse_repaint_mode(value, repaint.mode)) {
            fprintf(stderr, "[music-cli] --repaint-mode must be conservative|balanced|aggressive\n");
            return false;
        }
    }
    plan.emplace_back(std::move(repaint));
    return true;
}

static void parse_standalone_flow(
        int argc, char ** argv,
        const tts_cpp::acestep::GenerateParams & params,
        std::vector<tts_cpp::acestep::AudioEditParams> & plan) {
    using namespace tts_cpp::acestep;
    const char * source_caption = arg_val(argc, argv, kFlowSourceCaptionOption);
    if (!source_caption) return;
    FlowEditParams flow;
    flow.source_caption = source_caption;
    if (const char * value = arg_val(argc, argv, kFlowSourceLyricsOption))
        flow.source_lyrics = value;
    flow.target_caption = params.caption;
    flow.target_lyrics = params.lyrics;
    if (const char * value = arg_val(argc, argv, kFlowMinimumOption))
        flow.n_min = (float) atof(value);
    if (const char * value = arg_val(argc, argv, kFlowMaximumOption))
        flow.n_max = (float) atof(value);
    if (const char * value = arg_val(argc, argv, kFlowAverageOption))
        flow.n_avg = atoi(value);
    plan.emplace_back(std::move(flow));
}

static bool parse_edit_flags(int argc, char ** argv,
                             tts_cpp::acestep::GenerateParams & params) {
    const bool standalone_repaint = has_standalone_repaint_flags(argc, argv);
    const bool standalone_flow = has_standalone_flow_flags(argc, argv);
    if (const char * edit_plan_path = arg_val(argc, argv, kEditPlanOption)) {
        if (standalone_repaint || standalone_flow) {
            fprintf(stderr, "[music-cli] --edit-plan cannot be combined with standalone edit flags\n");
            return false;
        }
        std::string error;
        if (!load_edit_plan(edit_plan_path, params.edit_plan, error)) {
            fprintf(stderr, "[music-cli] edit plan failed: %s\n", error.c_str());
            return false;
        }
        return true;
    }
    if (!parse_standalone_repaint(argc, argv, params.edit_plan)) return false;
    parse_standalone_flow(argc, argv, params, params.edit_plan);
    return true;
}

static bool validate_edit_configuration(
        const tts_cpp::acestep::GenerateParams & params) {
    if (params.edit_plan.empty()) return true;
    if (params.source_audio.empty()) {
        fprintf(stderr, "[music-cli] editing requires --src-audio <48-kHz PCM16 WAV>\n");
        return false;
    }
    if (params.task_type != kTextToMusicTask) {
        fprintf(stderr, "[music-cli] editing cannot be combined with --task %s\n",
                params.task_type.c_str());
        return false;
    }
    fprintf(stderr, "[music-cli] edit plan: %zu ordered operation(s)\n",
            params.edit_plan.size());
    return true;
}

static std::vector<float> wav_read(const char * path, int * frames, int * rate) {
    tts_cpp::acestep::WavReadResult result = tts_cpp::acestep::load_pcm16_wav(path);
    if (!result.error.empty()) {
        fprintf(stderr, "[music-cli] %s: %s\n", path, result.error.c_str());
        return {};
    }
    *frames = result.frames;
    *rate   = result.sample_rate;
    return std::move(result.pcm);
}

static bool load_source_audio(int argc, char ** argv,
                              tts_cpp::acestep::GenerateParams & params) {
    const char * source_path = arg_val(argc, argv, kSourceAudioOption);
    if (!source_path) return true;
    int source_frames = 0;
    int source_rate = 0;
    params.source_audio = wav_read(source_path, &source_frames, &source_rate);
    if (params.source_audio.empty()) return false;
    if (source_rate != kRequiredAudioSampleRate) {
        fprintf(stderr, "[music-cli] source WAV is %d Hz; convert it to 48 kHz before generation\n",
                source_rate);
        return false;
    }
    fprintf(stderr, "[music-cli] source audio: %s (%.2fs, 48 kHz stereo)\n", source_path,
            (float) source_frames / kRequiredAudioSampleRate);
    return true;
}

static bool should_normalize_output(
        int argc, char ** argv,
        const tts_cpp::acestep::GenerateParams & params) {
    return params.edit_plan.empty() || arg_flag(argc, argv, kNormalizeOption);
}

static void wav_write(const char * path, const std::vector<float> & pcm,
                      int frames, int rate, bool normalize) {
    float peak = 1e-9f;
    for (int i = 0; i < frames * 2; i++) peak = std::fmax(peak, std::fabs(pcm[i]));
    float  gain = normalize ? 0.9f / peak : 1.0f;
    FILE * f    = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    const int ch = 2, bits = 16;
    uint32_t  db  = (uint32_t) frames * ch * (bits / 8);
    auto      w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto      w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w32(36 + db); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(ch); w32((uint32_t) rate);
    w32((uint32_t) rate * ch * (bits / 8)); w16(ch * (bits / 8)); w16(bits);
    fwrite("data", 1, 4, f); w32(db);
    for (int i = 0; i < frames * 2; i++) {
        float v = pcm[i] * gain * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        w16((uint16_t) (int16_t) lrintf(v));
    }
    fclose(f);
    fprintf(stderr, "[music-cli] wrote %s: %d frames, %.2fs @ %d Hz stereo\n", path, frames,
            (float) frames / rate, rate);
}

int main(int argc, char ** argv) {
    using namespace tts_cpp::acestep;

    EngineOptions o;
    o.verbose = true;
    if (arg_val(argc, argv, "--models")) o.models_dir = arg_val(argc, argv, "--models");
    if (arg_val(argc, argv, "--dit"))    o.dit_model_path = arg_val(argc, argv, "--dit");
    if (arg_val(argc, argv, "--lm"))     o.lm_model_path = arg_val(argc, argv, "--lm");
    if (arg_val(argc, argv, "--text"))   o.text_enc_model_path = arg_val(argc, argv, "--text");
    if (arg_val(argc, argv, "--vae"))    o.vae_model_path = arg_val(argc, argv, "--vae");
    // Offer every stage to a GPU backend when one is available. Engine::create
    // selects validated Vulkan/Metal or Adreno 700+ OpenCL where possible; DiT,
    // VAE and encoders use the selected GPU, while the LM and FSQ detokenizer
    // apply their independent backend allowlists and generic CPU fallback.
    if (arg_flag(argc, argv, "--gpu"))   o.n_gpu_layers = 99;
    if (arg_val(argc, argv, "--threads")) o.n_threads = atoi(arg_val(argc, argv, "--threads"));
    // Required wherever ggml ships its backends as dlopen'd MODULE .so files
    // (GGML_BACKEND_DL, i.e. every Android/arm64 build): without it the registry is
    // empty and even the CPU backend fails to init.
    if (arg_val(argc, argv, "--backends-dir")) o.backends_dir = arg_val(argc, argv, "--backends-dir");
    // Parity aid: write one .bin per stage so a CPU/GPU divergence can be traced
    // to the stage that introduces it rather than inferred from the final WAV.
    if (arg_val(argc, argv, "--dump-stages")) o.dump_stages_dir = arg_val(argc, argv, "--dump-stages");

    if (o.models_dir.empty() && o.dit_model_path.empty()) {
        fprintf(stderr,
                "usage: music-cli --models <dir> [--out song.wav] [--dur 8] [--seed 42]\n"
                "   or: music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf\n"
                "  prompt:  [--caption \"...\"] [--lyrics \"...\"] [--bpm 128] [--key \"C major\"]\n"
                "           [--tsig 4/4] [--lang en] [--req request.json]\n"
                "  audio:   [--ref-audio <48-kHz PCM16 WAV>]  (timbre reference)\n"
                "           [--src-audio <48-kHz PCM16 WAV>]  (cover source structure)\n"
                "           [--task text2music|cover-nofsq]   (default text2music)\n"
                "           [--cover-strength F] [--cover-noise F]  (cover defaults 1.0 / 0.0)\n"
                "  editing: [--repaint-start SEC] [--repaint-end SEC|-1]\n"
                "           [--repaint-mode conservative|balanced|aggressive]\n"
                "           [--repaint-strength 0..1]\n"
                "           [--flow-source-caption TEXT] [--flow-source-lyrics TEXT]\n"
                "           [--flow-n-min F] [--flow-n-max F] [--flow-n-avg N]\n"
                "           [--edit-plan plan.json]  (ordered/repeated operations)\n"
                "           plan.json: {\"operations\":[{\"type\":\"repaint\",...},\n"
                "             {\"type\":\"flow-edit\",\"source_caption\":\"...\",\n"
                "              \"target_caption\":\"...\",\"n_min\":0,\"n_max\":1,\"n_avg\":1}]}\n"
                "  sampler: [--steps N] [--shift F]  (default: auto from the DiT variant,\n"
                "           turbo 8 / 3.0, base and sft 50 / 1.0)\n"
                "           [--no-dcw]  (Haar DCW double mode is enabled by default)\n"
                "  output:  [--normalize]  (peak-normalize edit output before PCM quantization)\n"
                "           [--temp 0.85] [--cfg 2.0] [--topp 0.9] [--topk 0 (off)]\n"
                "           [--no-phase1]  (values shown are the defaults)\n"
                "  backend: [--gpu] [--threads N] [--backends-dir <dir>]\n"
                "           [--dump-stages <existing dir>]\n");
        return 1;
    }

    GenerateParams p;
    p.caption         = arg_val(argc, argv, "--caption") ? arg_val(argc, argv, "--caption")
                                                         : "Upbeat pop rock with driving electric guitars, punchy drums and a catchy hook";
    p.lyrics          = arg_val(argc, argv, "--lyrics") ? arg_val(argc, argv, "--lyrics") : "[Instrumental]";
    p.duration        = arg_val(argc, argv, "--dur")   ? (float) atof(arg_val(argc, argv, "--dur"))   : 8.0f;
    p.inference_steps = arg_val(argc, argv, "--steps") ? atoi(arg_val(argc, argv, "--steps"))         : 0;
    p.shift           = arg_val(argc, argv, "--shift") ? (float) atof(arg_val(argc, argv, "--shift")) : 0.0f;
    p.seed            = arg_val(argc, argv, "--seed")  ? strtoll(arg_val(argc, argv, "--seed"), nullptr, 10) : 42;
    if (arg_val(argc, argv, "--bpm"))  p.bpm = atoi(arg_val(argc, argv, "--bpm"));
    if (arg_val(argc, argv, "--key"))  p.keyscale = arg_val(argc, argv, "--key");
    if (arg_val(argc, argv, "--tsig")) p.timesignature = arg_val(argc, argv, "--tsig");
    if (arg_val(argc, argv, "--lang")) p.vocal_language = arg_val(argc, argv, "--lang");
    if (arg_val(argc, argv, "--temp")) p.lm_temperature = (float) atof(arg_val(argc, argv, "--temp"));
    if (arg_val(argc, argv, "--cfg"))  p.lm_cfg_scale = (float) atof(arg_val(argc, argv, "--cfg"));
    if (arg_val(argc, argv, "--topk")) p.lm_top_k = atoi(arg_val(argc, argv, "--topk"));
    if (arg_val(argc, argv, "--topp")) p.lm_top_p = (float) atof(arg_val(argc, argv, "--topp"));
    if (arg_flag(argc, argv, "--no-phase1")) p.lm_phase1 = false;
    if (arg_flag(argc, argv, "--no-dcw")) p.dcw_enabled = false;

    // --req <json>: load caption/lyrics/metas and (if present) audio_codes to
    // bypass our LM — used for parity against acestep.cpp's ace-lm output.
    if (arg_val(argc, argv, "--req")) {
        std::string j = read_file(arg_val(argc, argv, "--req"));
        if (j.empty()) { fprintf(stderr, "[music-cli] cannot read --req json\n"); return 1; }
        std::string v;
        if (json_field(j, "caption", v)) p.caption = v;
        if (json_field(j, "lyrics", v)) p.lyrics = v;
        if (json_field(j, "keyscale", v)) p.keyscale = v;
        if (json_field(j, "timesignature", v)) p.timesignature = v;
        if (json_field(j, "vocal_language", v)) p.vocal_language = v;
        if (json_field(j, "bpm", v)) p.bpm = atoi(v.c_str());
        if (json_field(j, "duration", v)) p.duration = (float) atof(v.c_str());
        if (json_field(j, "shift", v)) p.shift = (float) atof(v.c_str());
        if (json_field(j, "inference_steps", v)) p.inference_steps = atoi(v.c_str());
        if (json_field(j, "seed", v)) p.seed = strtoll(v.c_str(), nullptr, 10);
        if (json_field(j, "dcw_enabled", v)) p.dcw_enabled = v != "false" && v != "0";
        if (json_field(j, "dcw_scaler", v)) p.dcw_scaler = (float) atof(v.c_str());
        if (json_field(j, "dcw_high_scaler", v)) p.dcw_high_scaler = (float) atof(v.c_str());
        if (json_field(j, "task_type", v)) p.task_type = v;
        if (json_field(j, "audio_cover_strength", v)) p.audio_cover_strength = (float) atof(v.c_str());
        if (json_field(j, "cover_noise_strength", v)) p.cover_noise_strength = (float) atof(v.c_str());
        if (json_field(j, "audio_codes", v) && !v.empty()) {
            size_t start = 0;
            while (start < v.size()) {
                size_t comma = v.find(',', start);
                std::string tok = v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!tok.empty()) p.audio_codes.push_back(atoi(tok.c_str()));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            fprintf(stderr, "[music-cli] --req: %zu pre-supplied audio codes (LM bypass)\n", p.audio_codes.size());
        }
    }

    if (const char * reference_path = arg_val(argc, argv, "--ref-audio")) {
        int reference_frames = 0;
        int reference_rate   = 0;
        p.reference_audio = wav_read(reference_path, &reference_frames, &reference_rate);
        if (p.reference_audio.empty()) return 1;
        if (reference_rate != 48000) {
            fprintf(stderr, "[music-cli] reference WAV is %d Hz; convert it to 48 kHz before generation\n",
                    reference_rate);
            return 1;
        }
        fprintf(stderr, "[music-cli] reference audio: %s (%.2fs, 48 kHz stereo)\n", reference_path,
                (float) reference_frames / 48000.0f);
    }

    if (!load_source_audio(argc, argv, p)) return 1;

    if (const char * task = arg_val(argc, argv, "--task")) p.task_type = task;
    if (arg_val(argc, argv, "--cover-strength"))
        p.audio_cover_strength = (float) atof(arg_val(argc, argv, "--cover-strength"));
    if (arg_val(argc, argv, "--cover-noise"))
        p.cover_noise_strength = (float) atof(arg_val(argc, argv, "--cover-noise"));

    if (!parse_edit_flags(argc, argv, p)) return 1;
    if (!validate_edit_configuration(p)) return 1;

    const char * out_path = arg_val(argc, argv, "--out") ? arg_val(argc, argv, "--out") : "music_out.wav";

    std::unique_ptr<Engine> eng;
    try {
        eng = Engine::create(o);
    } catch (const std::exception & e) {
        fprintf(stderr, "[music-cli] engine create failed: %s\n", e.what());
        return 1;
    }

    auto t_start = std::chrono::steady_clock::now();
    auto t_vae0  = t_start;
    auto progress = [&](const std::string & stage, int step, int total) -> bool {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - t_start).count();
        if (stage == "vae" && step == 0) t_vae0 = now;
        if (stage == "vae" && step == total) {
            double vae_ms = std::chrono::duration<double, std::milli>(now - t_vae0).count();
            fprintf(stderr, "[music-cli] %-4s %d/%d  (t=%.0fms) VAE render=%.0fms\n",
                    stage.c_str(), step, total, ms, vae_ms);
        } else {
            fprintf(stderr, "[music-cli] %-4s %d/%d  (t=%.0fms)\n", stage.c_str(), step, total, ms);
        }
        return true;
    };

    GenerateResult r;
    try {
        r = eng->generate(p, progress);
    } catch (const std::exception & e) {
        fprintf(stderr, "[music-cli] generate failed: %s\n", e.what());
        return 1;
    }

    if (r.pcm.empty()) { fprintf(stderr, "[music-cli] no audio (cancelled?)\n"); return 1; }

    int frames = (int) (r.pcm.size() / 2);
    fprintf(stderr, "[music-cli] generated %d codes, seed=%lld, %d frames (%.2fs)\n", r.metadata.n_codes,
            r.metadata.seed, frames, (float) frames / r.sample_rate);
    wav_write(out_path, r.pcm, frames, r.sample_rate,
              should_normalize_output(argc, argv, p));
    return 0;
}
