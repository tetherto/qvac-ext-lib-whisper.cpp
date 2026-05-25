#include "audio.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace qwen::gguf {

namespace {

uint16_t read_u16_le(const uint8_t * p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32_le(const uint8_t * p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

void resample_naive(const std::vector<float> & in, int src_rate, int dst_rate,
                    std::vector<float> & out) {
    if (src_rate == dst_rate) {
        out = in;
        return;
    }
    const double ratio = static_cast<double>(src_rate) / dst_rate;
    const size_t n_out  = static_cast<size_t>(in.size() / ratio);
    out.resize(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double src_idx = i * ratio;
        const size_t k       = static_cast<size_t>(src_idx);
        const double frac    = src_idx - k;
        const float  a       = in[std::min(k, in.size() - 1)];
        const float  b       = in[std::min(k + 1, in.size() - 1)];
        out[i] = static_cast<float>(a * (1.0 - frac) + b * frac);
    }
}

void parse_wav_pcm16(const std::vector<uint8_t> & file, AudioSamples & out) {
    if (file.size() < 44) {
        throw std::runtime_error("qwen::gguf::load_wav_mono16: file too small to be WAV");
    }
    if (std::memcmp(file.data(), "RIFF", 4) != 0 || std::memcmp(file.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("qwen::gguf::load_wav_mono16: not a RIFF/WAVE file");
    }
    size_t off = 12;
    uint16_t audio_format    = 0;
    uint16_t channels        = 0;
    uint32_t sample_rate     = 0;
    uint16_t bits_per_sample = 0;
    const uint8_t * pcm_data = nullptr;
    uint32_t        pcm_size = 0;
    while (off + 8 <= file.size()) {
        const char *   tag = reinterpret_cast<const char *>(&file[off]);
        const uint32_t sz  = read_u32_le(&file[off + 4]);
        if (std::memcmp(tag, "fmt ", 4) == 0) {
            if (off + 8 + 16 > file.size()) break;
            audio_format    = read_u16_le(&file[off + 8]);
            channels        = read_u16_le(&file[off + 8 + 2]);
            sample_rate     = read_u32_le(&file[off + 8 + 4]);
            bits_per_sample = read_u16_le(&file[off + 8 + 14]);
        } else if (std::memcmp(tag, "data", 4) == 0) {
            pcm_data = &file[off + 8];
            pcm_size = sz;
            break;
        }
        off += 8 + sz;
    }
    if (audio_format != 1 || bits_per_sample != 16 || pcm_data == nullptr || channels < 1) {
        throw std::runtime_error("qwen::gguf::load_wav_mono16: only mono/stereo PCM16 WAV is supported");
    }
    const int          n_frames = static_cast<int>(pcm_size / (channels * 2));
    std::vector<float> mono(n_frames);
    const int16_t * src = reinterpret_cast<const int16_t *>(pcm_data);
    for (int i = 0; i < n_frames; ++i) {
        int32_t sum = 0;
        for (int c = 0; c < channels; ++c) {
            sum += src[i * channels + c];
        }
        mono[i] = static_cast<float>(sum) / (channels * 32768.0f);
    }
    if (static_cast<int>(sample_rate) != SAMPLE_RATE) {
        std::vector<float> resampled;
        resample_naive(mono, sample_rate, SAMPLE_RATE, resampled);
        out.data        = std::move(resampled);
        out.sample_rate = SAMPLE_RATE;
    } else {
        out.data        = std::move(mono);
        out.sample_rate = SAMPLE_RATE;
    }
}

void build_hann_window(std::vector<float> & w) {
    w.resize(WIN_LENGTH);
    for (int i = 0; i < WIN_LENGTH; ++i) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / WIN_LENGTH));
    }
}

float hertz_to_mel(float hz) {
    constexpr float min_log_hz    = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float     log_step      = std::log(6.4f) / 27.0f;
    float           mel           = 3.0f * hz / 200.0f;
    if (hz >= min_log_hz) {
        mel = min_log_mel + std::log(hz / min_log_hz) * (1.0f / log_step);
    }
    return mel;
}

float mel_to_hertz(float mel) {
    constexpr float min_log_hz    = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float     log_step      = std::log(6.4f) / 27.0f;
    float           hz            = 200.0f * mel / 3.0f;
    if (mel >= min_log_mel) {
        hz = min_log_hz * std::exp(log_step * (mel - min_log_mel));
    }
    return hz;
}

std::vector<float> build_mel_filters() {
    constexpr int   N_FREQ = N_FFT / 2 + 1;
    std::vector<float> filters(N_MEL * N_FREQ, 0.0f);
    std::vector<float> fft_freqs(N_FREQ);
    for (int i = 0; i < N_FREQ; ++i) {
        fft_freqs[i] = static_cast<float>(i) * (SAMPLE_RATE / 2.0f) / static_cast<float>(N_FREQ - 1);
    }
    const float mel_min = hertz_to_mel(0.0f);
    const float mel_max = hertz_to_mel(SAMPLE_RATE / 2.0f);
    std::vector<float> filter_freqs(N_MEL + 2);
    for (int i = 0; i < N_MEL + 2; ++i) {
        const float m = mel_min + (mel_max - mel_min) * i / (N_MEL + 1);
        filter_freqs[i] = mel_to_hertz(m);
    }
    std::vector<float> filter_diff(N_MEL + 1);
    for (int i = 0; i < N_MEL + 1; ++i) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
        if (filter_diff[i] == 0.0f) filter_diff[i] = 1e-6f;
    }
    for (int m = 0; m < N_MEL; ++m) {
        const float enorm = 2.0f / (filter_freqs[m + 2] - filter_freqs[m]);
        for (int f = 0; f < N_FREQ; ++f) {
            const float down = (fft_freqs[f] - filter_freqs[m]) / filter_diff[m];
            const float up   = (filter_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            const float val  = std::max(0.0f, std::min(down, up));
            filters[m * N_FREQ + f] = val * enorm;
        }
    }
    return filters;
}

void reflect_pad(const float * samples, int n_samples, int pad, std::vector<float> & padded) {
    padded.assign(n_samples + 2 * pad, 0.0f);
    for (int i = 0; i < pad; ++i) {
        const int src = pad - i;
        padded[i] = (src < n_samples) ? samples[src] : 0.0f;
    }
    std::memcpy(padded.data() + pad, samples, n_samples * sizeof(float));
    for (int i = 0; i < pad; ++i) {
        const int src = n_samples - 2 - i;
        padded[pad + n_samples + i] = (src >= 0) ? samples[src] : 0.0f;
    }
}

}

AudioSamples load_wav_mono16(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("qwen::gguf::load_wav_mono16: cannot open '" + path + "'");
    }
    in.seekg(0, std::ios::end);
    const std::streamsize n = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(buf.data()), n);
    AudioSamples out;
    parse_wav_pcm16(buf, out);
    return out;
}

MelSpectrogram log_mel_spectrogram(const float * samples, int n_samples) {
    constexpr int N_FREQ = N_FFT / 2 + 1;

    std::vector<float> padded;
    reflect_pad(samples, n_samples, N_FFT / 2, padded);
    const int padded_len    = static_cast<int>(padded.size());
    const int n_frames_full = (padded_len - N_FFT) / HOP_LENGTH + 1;
    const int n_frames      = n_frames_full - 1;
    if (n_frames <= 0) {
        throw std::runtime_error("qwen::gguf::log_mel_spectrogram: audio too short");
    }

    std::vector<float> window;
    build_hann_window(window);

    std::vector<float> dft_cos(N_FREQ * N_FFT);
    std::vector<float> dft_sin(N_FREQ * N_FFT);
    for (int k = 0; k < N_FREQ; ++k) {
        for (int n = 0; n < N_FFT; ++n) {
            const float angle = 2.0f * static_cast<float>(M_PI) * k * n / N_FFT;
            dft_cos[k * N_FFT + n] = std::cos(angle);
            dft_sin[k * N_FFT + n] = std::sin(angle);
        }
    }

    const auto filters = build_mel_filters();

    std::vector<float> mel_tmp(n_frames * N_MEL, 0.0f);
    std::vector<float> windowed(N_FFT);
    std::vector<float> power(N_FREQ);
    float              global_max = -1e30f;

    for (int t = 0; t < n_frames; ++t) {
        const int start = t * HOP_LENGTH;
        for (int i = 0; i < N_FFT; ++i) {
            windowed[i] = padded[start + i] * window[i];
        }
        for (int k = 0; k < N_FREQ; ++k) {
            float re = 0.0f;
            float im = 0.0f;
            const float * c = &dft_cos[k * N_FFT];
            const float * s = &dft_sin[k * N_FFT];
            for (int n = 0; n < N_FFT; ++n) {
                re += windowed[n] * c[n];
                im += windowed[n] * s[n];
            }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < N_MEL; ++m) {
            float sum = 0.0f;
            const float * f = &filters[m * N_FREQ];
            for (int k = 0; k < N_FREQ; ++k) sum += f[k] * power[k];
            if (sum < 1e-10f) sum = 1e-10f;
            const float val = std::log10(sum);
            mel_tmp[t * N_MEL + m] = val;
            if (val > global_max) global_max = val;
        }
    }

    MelSpectrogram out;
    out.n_mels   = N_MEL;
    out.n_frames = n_frames;
    out.data.assign(N_MEL * n_frames, 0.0f);
    const float floor_val = global_max - MEL_FLOOR_DB;
    for (int t = 0; t < n_frames; ++t) {
        for (int m = 0; m < N_MEL; ++m) {
            float v = mel_tmp[t * N_MEL + m];
            if (v < floor_val) v = floor_val;
            out.data[m * n_frames + t] = (v + MEL_BIAS) / MEL_SCALE;
        }
    }
    return out;
}

}
