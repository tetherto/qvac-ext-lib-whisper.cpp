#pragma once

#include "audiogen-cpp/acestep/engine.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace tts_cpp::acestep {

inline constexpr const char * TASK_TEXT2MUSIC  = "text2music";
inline constexpr const char * TASK_COVER       = "cover";
inline constexpr const char * TASK_COVER_NOFSQ = "cover-nofsq";

inline bool is_cover_task(const std::string & task) {
    return task == TASK_COVER || task == TASK_COVER_NOFSQ;
}

struct GenerateTask {
    std::string type;
    float       audio_cover_strength = 1.0f;
    float       cover_noise_strength = 0.0f;
};

inline float clamp_strength(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

inline constexpr const char * DEFAULT_VOCAL_LANGUAGE = "en";
inline constexpr const char * EDIT_VOCAL_LANGUAGE    = "unknown";
inline constexpr const char * INSTRUMENTAL_LYRICS    = "[Instrumental]";

// Simple mode keeps an unset language empty so the LM inspire pass picks it;
// otherwise the engine defaults apply before the prompt is built.
inline std::string resolve_prompt_language(const GenerateParams & params) {
    if (!params.vocal_language.empty()) return params.vocal_language;
    if (params.simple_mode) return {};
    return params.edit_plan.empty() ? DEFAULT_VOCAL_LANGUAGE : EDIT_VOCAL_LANGUAGE;
}

// Simple mode keeps unset lyrics empty so the LM inspire pass writes them
// (the explicit "[Instrumental]" request still flows through as the hint).
inline std::string resolve_prompt_lyrics(const GenerateParams & params) {
    if (!params.lyrics.empty()) return params.lyrics;
    if (params.simple_mode) return {};
    return INSTRUMENTAL_LYRICS;
}

inline std::string validate_simple_mode(const GenerateParams & params, const std::string & task_type) {
    if (!params.simple_mode) return {};
    if (params.caption.empty()) {
        return "acestep engine: simple_mode requires a caption query";
    }
    if (task_type != TASK_TEXT2MUSIC) {
        return "acestep engine: simple_mode supports only task 'text2music', got '" + task_type + "'";
    }
    if (!params.audio_codes.empty()) {
        return "acestep engine: simple_mode cannot take pre-supplied audio_codes";
    }
    if (!params.lyrics.empty() && params.lyrics != AUDIO_EDIT_DEFAULT_LYRICS) {
        return "acestep engine: simple_mode lyrics must be empty (LM writes them) or \"[Instrumental]\"";
    }
    if (!params.edit_plan.empty()) {
        return "acestep engine: simple_mode cannot be combined with edit_plan";
    }
    if (!params.lm_phase1) {
        return "acestep engine: simple_mode requires lm_phase1";
    }
    return {};
}

inline std::string resolve_generate_task(const GenerateParams & params, GenerateTask & task) {
    task.type = params.task_type.empty() ? TASK_TEXT2MUSIC : params.task_type;

    if (task.type != TASK_TEXT2MUSIC && task.type != TASK_COVER &&
        task.type != TASK_COVER_NOFSQ) {
        return "acestep engine: unsupported task_type '" + task.type +
               "' (expected text2music|cover|cover-nofsq)";
    }

    const std::string simple_mode_error = validate_simple_mode(params, task.type);
    if (!simple_mode_error.empty()) return simple_mode_error;

    if (!std::isfinite(params.audio_cover_strength)) {
        return "acestep engine: audio_cover_strength must be finite";
    }
    if (!std::isfinite(params.cover_noise_strength)) {
        return "acestep engine: cover_noise_strength must be finite";
    }

    task.audio_cover_strength = clamp_strength(params.audio_cover_strength);
    task.cover_noise_strength = clamp_strength(params.cover_noise_strength);

    if (!params.reference_audio.empty() && (params.reference_audio.size() & 1u) != 0) {
        return "acestep engine: reference_audio must be interleaved stereo";
    }

    if (!is_cover_task(task.type)) return {};

    if (params.source_audio.empty()) {
        return "acestep engine: task '" + task.type + "' requires source_audio";
    }
    if ((params.source_audio.size() & 1u) != 0) {
        return "acestep engine: source_audio must be interleaved stereo";
    }
    if (task.type == TASK_COVER) {
        return "acestep engine: task 'cover' is not implemented yet (needs FSQ tokenizer); use cover-nofsq";
    }
    return {};
}

inline bool needs_cover_conditioning_switch(const GenerateTask & task) {
    return is_cover_task(task.type) && task.audio_cover_strength < 1.0f;
}

// Step index where the DiT drops the source conditioning; -1 disables the switch.
inline int resolve_cover_switch_step(const GenerateTask & task, int num_steps) {
    if (!needs_cover_conditioning_switch(task)) return -1;
    return (int) ((float) num_steps * task.audio_cover_strength);
}

} // namespace tts_cpp::acestep
