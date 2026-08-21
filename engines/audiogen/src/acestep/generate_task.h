#pragma once

#include "audiogen-cpp/acestep/engine.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace tts_cpp::acestep {

inline constexpr const char * TASK_TEXT2MUSIC  = "text2music";
inline constexpr const char * TASK_COVER       = "cover";
inline constexpr const char * TASK_COVER_NOFSQ = "cover-nofsq";
inline constexpr const char * TASK_LEGO        = "lego";

inline constexpr const char * LEGO_TRACK_NAMES[] = {
    "vocals",     "backing_vocals", "drums", "bass", "guitar", "keyboard",
    "percussion", "strings",        "synth", "fx",   "brass",  "woodwinds",
};

inline bool is_cover_task(const std::string & task) {
    return task == TASK_COVER || task == TASK_COVER_NOFSQ;
}

inline bool is_lego_task(const std::string & task) {
    return task == TASK_LEGO;
}

inline bool is_source_task(const std::string & task) {
    return is_cover_task(task) || is_lego_task(task);
}

inline bool is_valid_lego_track(const std::string & track) {
    for (const char * name : LEGO_TRACK_NAMES) {
        if (track == name) return true;
    }
    return false;
}

struct GenerateTask {
    std::string type;
    std::string track;
    float       audio_cover_strength = 1.0f;
    float       cover_noise_strength = 0.0f;
};

inline float clamp_strength(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

inline constexpr float TURBO_GUIDANCE_SCALE    = 1.0f;
inline constexpr float STANDARD_GUIDANCE_SCALE = 7.0f;

inline std::string lego_model_error(bool is_turbo, bool is_sft) {
    if (is_turbo) {
        return "acestep engine: task 'lego' requires a base DiT (turbo does not support stem tasks)";
    }
    if (is_sft) {
        return "acestep engine: task 'lego' requires a base DiT (sft does not support stem tasks)";
    }
    return {};
}

// Turbo is guidance-distilled: CFG is untrained there, so explicit overrides
// clamp to 1.0. Base/sft default to the official 7.0 when unset.
inline float resolve_guidance_scale(float requested, bool is_turbo) {
    if (is_turbo) return TURBO_GUIDANCE_SCALE;
    return requested > 0.0f ? requested : STANDARD_GUIDANCE_SCALE;
}

// Haar DCW is a turbo-preset correction; the official base/sft preset disables it.
inline bool resolve_dcw_enabled(bool requested, bool is_turbo) {
    return requested && is_turbo;
}

inline std::string validate_lego_track(const std::string & track) {
    if (track.empty()) {
        return "acestep engine: task 'lego' requires a track name";
    }
    if (!is_valid_lego_track(track)) {
        return "acestep engine: unknown lego track '" + track +
               "' (expected one of vocals|backing_vocals|drums|bass|guitar|keyboard|"
               "percussion|strings|synth|fx|brass|woodwinds)";
    }
    return {};
}

inline std::string resolve_generate_task(const GenerateParams & params, GenerateTask & task) {
    task.type = params.task_type.empty() ? TASK_TEXT2MUSIC : params.task_type;

    if (task.type != TASK_TEXT2MUSIC && task.type != TASK_COVER &&
        task.type != TASK_COVER_NOFSQ && task.type != TASK_LEGO) {
        return "acestep engine: unsupported task_type '" + task.type +
               "' (expected text2music|cover|cover-nofsq|lego)";
    }

    if (!std::isfinite(params.audio_cover_strength)) {
        return "acestep engine: audio_cover_strength must be finite";
    }
    if (!std::isfinite(params.cover_noise_strength)) {
        return "acestep engine: cover_noise_strength must be finite";
    }

    task.track                = params.track;
    task.audio_cover_strength = clamp_strength(params.audio_cover_strength);
    task.cover_noise_strength = clamp_strength(params.cover_noise_strength);

    if (!params.reference_audio.empty() && (params.reference_audio.size() & 1u) != 0) {
        return "acestep engine: reference_audio must be interleaved stereo";
    }

    if (!is_source_task(task.type)) return {};

    if (params.source_audio.empty()) {
        return "acestep engine: task '" + task.type + "' requires source_audio";
    }
    if ((params.source_audio.size() & 1u) != 0) {
        return "acestep engine: source_audio must be interleaved stereo";
    }
    if (is_lego_task(task.type)) {
        return validate_lego_track(task.track);
    }
    if (task.type == TASK_COVER) {
        return "acestep engine: task 'cover' is not implemented yet (needs FSQ tokenizer); use cover-nofsq";
    }
    if (task.audio_cover_strength < 1.0f) {
        return "acestep engine: audio_cover_strength < 1 is not implemented yet for cover-nofsq";
    }
    return {};
}

} // namespace tts_cpp::acestep
