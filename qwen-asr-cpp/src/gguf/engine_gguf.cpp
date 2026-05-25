#include "engine_gguf.h"
#include "audio.h"
#include "model.h"
#include "model_loader.h"
#include "tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qwen::gguf {

namespace {

std::string format_load_summary(const ModelLoader & m) {
    const auto & h = m.hparams();
    std::ostringstream ss;
    ss << "qwen3-asr GGUF loaded: "
       << "enc[layers=" << h.enc_layers
       << " dim=" << h.enc_dim
       << " ff="  << h.enc_ff
       << " heads=" << h.enc_heads << "] "
       << "dec[layers=" << h.text_layers
       << " dim=" << h.text_dim
       << " ff="  << h.text_ff
       << " heads=" << h.text_heads
       << "/" << h.text_kv_heads
       << " head_dim=" << h.text_head_dim
       << " rope_base=" << h.text_rope_base
       << " rms_eps="  << h.text_rms_eps << "] "
       << "vocab=" << h.text_vocab
       << " tensors=" << m.tensors().size()
       << " weights=" << (m.total_bytes() / (1024 * 1024)) << " MiB";
    return ss.str();
}

void verify_tensor_table(const ModelLoader & m) {
    const auto required = expected_tensor_names(m.hparams());
    const auto missing  = m.missing(required);
    if (!missing.empty()) {
        std::ostringstream ss;
        ss << "qwen::gguf::Engine: model file is missing "
           << missing.size() << " required tensor(s); first: '"
           << missing.front() << "'";
        throw std::runtime_error(ss.str());
    }
}

void log_tokenizer_selftest(const Tokenizer & tok) {
    const std::string sample = "Hello world.";
    const auto ids   = tok.encode(sample, false);
    const auto round = tok.decode(ids, true);
    std::fprintf(stderr,
        "qwen-asr gguf tokenizer selftest: '%s' -> %zu ids -> '%s' (roundtrip %s)\n",
        sample.c_str(), ids.size(), round.c_str(),
        round == sample ? "OK" : "MISMATCH");
}

void log_encoder_summary(const EncoderOutput & enc) {
    float mn = enc.data.empty() ? 0.0f :  std::numeric_limits<float>::infinity();
    float mx = enc.data.empty() ? 0.0f : -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    for (float v : enc.data) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
    }
    const double mean = enc.data.empty() ? 0.0 : sum / enc.data.size();
    std::fprintf(stderr,
        "qwen-asr gguf encoder: seq_len=%d dim=%d range=[%.4f,%.4f] mean=%.4f\n",
        enc.seq_len, enc.dim, mn, mx, mean);
}

void dump_encoder_f32(const EncoderOutput & enc, const char * path) {
    std::FILE * f = std::fopen(path, "wb");
    if (f == nullptr) return;
    uint32_t ndim = 2;
    std::fwrite(&ndim, sizeof(ndim), 1, f);
    int32_t d0 = enc.seq_len;
    int32_t d1 = enc.dim;
    std::fwrite(&d0, sizeof(d0), 1, f);
    std::fwrite(&d1, sizeof(d1), 1, f);
    std::fwrite(enc.data.data(), sizeof(float), enc.data.size(), f);
    std::fclose(f);
    std::fprintf(stderr, "qwen-asr gguf encoder: dumped %s [%d,%d]\n", path, d0, d1);
}

void log_mel_summary(const MelSpectrogram & mel, int n_samples) {
    float mn = mel.data.empty() ? 0.0f :  std::numeric_limits<float>::infinity();
    float mx = mel.data.empty() ? 0.0f : -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    for (float v : mel.data) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
    }
    const double mean = mel.data.empty() ? 0.0 : sum / mel.data.size();
    std::fprintf(stderr,
        "qwen-asr gguf mel: samples=%d sample_rate=%d frames=%d mels=%d range=[%.4f,%.4f] mean=%.4f\n",
        n_samples, SAMPLE_RATE, mel.n_frames, mel.n_mels, mn, mx, mean);
}

class Engine final : public IEngine {
public:
    explicit Engine(const EngineOptions & opts) : opts_(opts) {
        if (opts_.model_path.empty()) {
            throw std::runtime_error(
                "qwen::gguf::Engine: EngineOptions.model_path is required (.gguf file)");
        }
        loader_ = std::make_unique<ModelLoader>(opts_.model_path);
        verify_tensor_table(*loader_);
        tokenizer_ = std::make_unique<Tokenizer>(loader_->vocab(), loader_->merges());
        tokenizer_->set_bos(static_cast<int32_t>(loader_->hparams().bos_token_id));
        tokenizer_->set_eos(static_cast<int32_t>(loader_->hparams().eos_token_id));
        model_     = std::make_unique<Model>(*loader_);
        if (opts_.verbose > 0) {
            std::fprintf(stderr, "%s\n", format_load_summary(*loader_).c_str());
            log_tokenizer_selftest(*tokenizer_);
        }
    }

    EngineResult transcribe(const std::string & wav_path) override {
        const auto wav = load_wav_mono16(wav_path);
        return transcribe_samples(wav.data.data(), static_cast<int>(wav.data.size()));
    }

    EngineResult transcribe_samples(const float * samples, int n_samples) override {
        const auto t_total_begin = std::chrono::steady_clock::now();
        auto mel = log_mel_spectrogram(samples, n_samples);
        if (const char * dump = std::getenv("QWEN_DUMP_MEL")) {
            std::FILE * f = std::fopen(dump, "wb");
            if (f) {
                uint32_t ndim = 2;
                std::fwrite(&ndim, sizeof(ndim), 1, f);
                int32_t d0 = mel.n_mels;
                int32_t d1 = mel.n_frames;
                std::fwrite(&d0, sizeof(d0), 1, f);
                std::fwrite(&d1, sizeof(d1), 1, f);
                std::fwrite(mel.data.data(), sizeof(float), mel.data.size(), f);
                std::fclose(f);
                std::fprintf(stderr, "qwen-asr gguf mel: dumped %s [%d,%d]\n", dump, d0, d1);
            }
        }
        if (const char * load_path = std::getenv("QWEN_LOAD_MEL")) {
            std::FILE * f = std::fopen(load_path, "rb");
            if (f) {
                uint32_t ndim = 0;
                std::fread(&ndim, sizeof(ndim), 1, f);
                int32_t d0 = 0, d1 = 0;
                std::fread(&d0, sizeof(d0), 1, f);
                std::fread(&d1, sizeof(d1), 1, f);
                std::vector<float> buf(static_cast<size_t>(d0) * d1);
                std::fread(buf.data(), sizeof(float), buf.size(), f);
                std::fclose(f);
                mel.n_mels   = d0;
                mel.n_frames = d1;
                mel.data     = std::move(buf);
                std::fprintf(stderr, "qwen-asr gguf mel: loaded %s [%d,%d]\n", load_path, d0, d1);
            }
        }
        if (opts_.verbose > 0) log_mel_summary(mel, n_samples);
        const int n_threads = opts_.n_threads > 0 ? opts_.n_threads : 4;
        const auto t_encode_begin = std::chrono::steady_clock::now();
        const auto enc_out = encode_audio(*model_, mel, n_threads, opts_.verbose);
        const auto t_encode_end = std::chrono::steady_clock::now();
        if (opts_.verbose > 0) log_encoder_summary(enc_out);
        if (const char * dump = std::getenv("QWEN_DUMP_ENCODER")) {
            dump_encoder_f32(enc_out, dump);
        }
        const auto t_decode_begin = std::chrono::steady_clock::now();
        auto result = run_greedy_decode(enc_out, n_threads);
        const auto t_decode_end = std::chrono::steady_clock::now();
        const auto t_total_end = std::chrono::steady_clock::now();
        result.encode_ms = std::chrono::duration<double, std::milli>(t_encode_end - t_encode_begin).count();
        result.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_begin).count();
        result.total_ms  = std::chrono::duration<double, std::milli>(t_total_end  - t_total_begin ).count();
        result.audio_ms  = static_cast<double>(n_samples) * 1000.0 / 16000.0;
        return result;
    }

    EngineResult run_greedy_decode(const EncoderOutput & enc_out, int n_threads) {
        const Hparams & h = loader_->hparams();
        const std::vector<int32_t> prefix_head = { 151644, 8948, 198 };
        const std::vector<int32_t> prefix_tail = { 151645, 198, 151644, 872, 198, static_cast<int32_t>(h.audio_start_id) };
        const std::vector<int32_t> suffix_base = { static_cast<int32_t>(h.audio_end_id), 151645, 198, 151644, 77091, 198 };

        std::vector<int32_t> tokens;
        tokens.reserve(prefix_head.size() + prefix_tail.size() + enc_out.seq_len + suffix_base.size() + 64);
        tokens.insert(tokens.end(), prefix_head.begin(), prefix_head.end());
        tokens.insert(tokens.end(), prefix_tail.begin(), prefix_tail.end());
        for (int i = 0; i < enc_out.seq_len; ++i) {
            tokens.push_back(static_cast<int32_t>(h.audio_token_id));
        }
        tokens.insert(tokens.end(), suffix_base.begin(), suffix_base.end());

        const int32_t asr_text_token = 151704;
        if (!opts_.language.empty()) {
            const std::string lang_text = "language " + opts_.language;
            const auto lang_ids = tokenizer_->encode(lang_text, false);
            tokens.insert(tokens.end(), lang_ids.begin(), lang_ids.end());
            tokens.push_back(asr_text_token);
        }

        const int ctx_max  = std::max<int>(2048, static_cast<int>(tokens.size() + 512));
        const int max_new  = opts_.max_new_tokens > 0 ? opts_.max_new_tokens : 256;
        const int eos_id   = static_cast<int32_t>(h.eos_token_id);
        const int audio_id = static_cast<int32_t>(h.audio_token_id);
        const int asr_text_id = asr_text_token;

        DecoderState state;
        init_decoder_state(state, *model_, ctx_max);

        std::vector<int32_t> positions(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) positions[i] = static_cast<int32_t>(i);

        if (opts_.verbose > 0) {
            std::fprintf(stderr, "qwen-asr gguf decoder: prefill %zu tokens (audio=%d)\n",
                tokens.size(), enc_out.seq_len);
        }
        auto step0 = decoder_step(*model_, state, tokens, positions, enc_out, audio_id, n_threads, true);

        std::vector<int32_t> generated;
        int32_t next = greedy_argmax(step0.logits.data(), step0.vocab_size);
        generated.push_back(next);
        if (opts_.verbose > 0) {
            std::fprintf(stderr, "qwen-asr gguf decoder: first token = %d ('%s')\n",
                next, tokenizer_->id_to_piece(next).c_str());
        }

        for (int s = 1; s < max_new; ++s) {
            if (cancelled_.load(std::memory_order_relaxed)) break;
            if (next == eos_id) break;
            const int pos = static_cast<int>(tokens.size()) + s - 1;
            std::vector<int32_t> step_in  = { next };
            std::vector<int32_t> step_pos = { pos };
            auto step = decoder_step(*model_, state, step_in, step_pos, enc_out, audio_id, n_threads, true);
            next = greedy_argmax(step.logits.data(), step.vocab_size);
            generated.push_back(next);
            if (opts_.verbose > 1) {
                std::fprintf(stderr, "qwen-asr gguf decoder: step %d token=%d ('%s')\n",
                    s, next, tokenizer_->id_to_piece(next).c_str());
            }
            if (token_cb_) token_cb_(tokenizer_->id_to_piece(next));
        }

        if (!generated.empty() && generated.back() == eos_id) generated.pop_back();

        int asr_pos = -1;
        for (size_t i = 0; i < generated.size(); ++i) {
            if (generated[i] == asr_text_id) { asr_pos = static_cast<int>(i); break; }
        }
        const std::vector<int32_t> text_ids(generated.begin() + (asr_pos >= 0 ? asr_pos + 1 : 0),
                                            generated.end());
        const std::string text = tokenizer_->decode(text_ids, true);

        EngineResult r;
        r.text        = text;
        r.text_tokens = static_cast<int>(text_ids.size());
        return r;
    }

    void set_token_callback(TokenCallback cb) override {
        token_cb_ = std::move(cb);
    }

    void cancel() override {
        cancelled_.store(true, std::memory_order_relaxed);
    }

    const EngineOptions & options() const override { return opts_; }

    Backend backend() const override { return Backend::GGUF; }

private:
    EngineOptions                opts_;
    std::unique_ptr<ModelLoader> loader_;
    std::unique_ptr<Tokenizer>   tokenizer_;
    std::unique_ptr<Model>       model_;
    TokenCallback                token_cb_;
    std::atomic_bool             cancelled_{false};
};

}

std::unique_ptr<IEngine> make_engine(const EngineOptions & opts) {
    return std::unique_ptr<IEngine>(new Engine(opts));
}

}
