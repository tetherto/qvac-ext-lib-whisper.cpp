#pragma once

// Pure helpers shared by the fit projector (parakeet_fit.cpp) and the
// parakeet-fit-params CLI (fit_main.cpp), kept header-only and dependency-free
// so test/test_fit_cli.cpp can exercise them model-free.
//
// The saturating arithmetic exists for one reason: a preflight's verdict must
// only ever be wrong in the strict direction. Any overflow surfaces as a
// larger requirement (DOES-NOT-FIT), never wraps into a false FITS.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace parakeet {
namespace fitutil {

inline uint64_t sat_add(uint64_t a, uint64_t b) {
    const uint64_t s = a + b;
    return s < a ? std::numeric_limits<uint64_t>::max() : s;
}

inline uint64_t sat_mul(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > std::numeric_limits<uint64_t>::max() / b) {
        return std::numeric_limits<uint64_t>::max();
    }
    return a * b;
}

// Negative inputs collapse to 0; too-large, infinite, and NaN inputs all
// saturate to max -- an unrepresentable requirement must read as "does not
// fit", never as "needs nothing".
inline uint64_t sat_u64_from_double(double d) {
    if (std::isnan(d)) return std::numeric_limits<uint64_t>::max();
    if (d <= 0.0) return 0;
    if (d >= (double) std::numeric_limits<uint64_t>::max()) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (uint64_t) d;
}

// ── Strict CLI parsers ─────────────────────────────────────────────────────
// A preflight must fail loudly on garbage input, not coerce it (atoi/atof
// return 0 on junk, silently changing which device or workload is projected).

inline bool parse_f32_positive(const char * s, float & out) {
    if (!s) return false;
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !std::isfinite(v) || v <= 0.0) return false;
    out = (float) v;
    return true;
}

inline bool parse_u64(const char * s, uint64_t & out) {
    if (!s || *s == '-') return false;
    char * end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = (uint64_t) v;
    return true;
}

inline bool parse_i32(const char * s, int & out) {
    if (!s) return false;
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' ||
        v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = (int) v;
    return true;
}

// Saturating MiB -> bytes: an absurd margin makes the verdict stricter,
// never overflows into a false FITS.
inline uint64_t margin_mib_to_bytes(uint64_t mib) {
    return sat_mul(mib, 1024ull * 1024ull);
}

// Escape a string for embedding in a JSON string literal. model_variant comes
// from GGUF metadata and device_name from the backend/driver, so neither is
// trusted to be JSON-clean; @qvac/model-fit JSON.parses this output.
inline std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char ch : s) {
        const unsigned char c = (unsigned char) ch;
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    return out;
}

}  // namespace fitutil
}  // namespace parakeet
