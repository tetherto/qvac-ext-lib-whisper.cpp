#pragma once
// English number normalization for TTS prompts: expands ASCII digit runs to
// words (cardinals with thousands separators, decimals, ordinals).

#include <string>

namespace tts_cpp {
namespace parler {
namespace detail {

// Every byte outside a recognized number token passes through unchanged
// (UTF-8 safe); output contains no ASCII digits, so the pass is idempotent.
std::string normalize_numbers_en(const std::string & text);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
