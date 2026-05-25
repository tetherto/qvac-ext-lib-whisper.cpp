#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen::gguf {

constexpr int   SAMPLE_RATE   = 16000;
constexpr int   N_FFT         = 400;
constexpr int   HOP_LENGTH    = 160;
constexpr int   WIN_LENGTH    = 400;
constexpr int   N_MEL         = 128;
constexpr float MEL_FLOOR_DB  = 8.0f;
constexpr float MEL_BIAS      = 4.0f;
constexpr float MEL_SCALE     = 4.0f;

struct AudioSamples {
    std::vector<float> data;
    int                sample_rate = SAMPLE_RATE;
};

AudioSamples load_wav_mono16(const std::string & path);

struct MelSpectrogram {
    std::vector<float> data;
    int                n_mels   = 0;
    int                n_frames = 0;
};

MelSpectrogram log_mel_spectrogram(const float * samples, int n_samples);

}
