// Model-free unit tests for the acestep-fit-params CLI layer
// (src/acestep/fit_util.h + acestep_fit_cli_main argument handling). These
// cover the pieces the fixture-gated parity test (test_fit_params.cpp) cannot:
// the strict numeric parsers, the saturating arithmetic the verdict depends
// on, the JSON string escaper the @qvac/model-fit SDK will JSON.parse, and the
// CLI's exit-code contract for garbage input (exit 2 = Error, never a
// coerced-to-0 flag).
//
// Exit 0 on success; non-zero (with a FAIL line per broken invariant) otherwise.

#include "audiogen-cpp/acestep/fit.h"

#include "fit_util.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

extern "C" int acestep_fit_cli_main(int argc, char ** argv);

using namespace tts_cpp::acestep::fitutil;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

// argv-level driver: acestep_fit_cli_main takes mutable char**.
int run_cli(std::vector<std::string> args) {
    args.insert(args.begin(), "acestep-fit-params");
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (std::string & a : args) argv.push_back(a.data());
    return acestep_fit_cli_main((int) argv.size(), argv.data());
}

constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

}  // namespace

int main() {
    // ── Strict parsers: garbage fails, never coerces ────────────────────────
    {
        int i = 7;
        expect(parse_i32("42", i) && i == 42,   "parse_i32('42')");
        expect(parse_i32("-3", i) && i == -3,   "parse_i32('-3')");
        expect(!parse_i32("1x", i),             "parse_i32 accepted trailing garbage '1x'");
        expect(!parse_i32("", i),               "parse_i32 accepted empty string");
        expect(!parse_i32("99999999999999", i), "parse_i32 accepted out-of-range value");

        uint64_t u = 7;
        expect(parse_u64("0", u) && u == 0,     "parse_u64('0')");
        expect(parse_u64("18446744073709551615", u) && u == U64_MAX,
               "parse_u64(UINT64_MAX)");
        expect(!parse_u64("-1", u),             "parse_u64 accepted '-1'");
        expect(!parse_u64("12abc", u),          "parse_u64 accepted trailing garbage");

        float f = 7.0f;
        expect(parse_f32_positive("2.5", f) && f == 2.5f, "parse_f32_positive('2.5')");
        expect(!parse_f32_positive("abc", f),  "parse_f32_positive accepted 'abc'");
        expect(!parse_f32_positive("0", f),    "parse_f32_positive accepted '0'");
        expect(!parse_f32_positive("-5", f),   "parse_f32_positive accepted '-5'");
        expect(!parse_f32_positive("inf", f),  "parse_f32_positive accepted 'inf'");
        expect(!parse_f32_positive("1e9999", f),
               "parse_f32_positive accepted an overflowing literal");
        expect(!parse_f32_positive("10x", f),  "parse_f32_positive accepted trailing garbage");

        expect(parse_f32_nonnegative("0", f) && f == 0.0f, "parse_f32_nonnegative('0')");
        expect(parse_f32_nonnegative("7.5", f) && f == 7.5f, "parse_f32_nonnegative('7.5')");
        expect(!parse_f32_nonnegative("-0.5", f), "parse_f32_nonnegative accepted '-0.5'");
        expect(!parse_f32_nonnegative("nan", f),  "parse_f32_nonnegative accepted 'nan'");
    }

    // ── Saturating arithmetic: overflow must widen, never wrap ─────────────
    {
        expect(sat_add(1, 2) == 3,                       "sat_add(1,2)");
        expect(sat_add(U64_MAX, 1) == U64_MAX,           "sat_add saturates");
        expect(sat_add(U64_MAX, U64_MAX) == U64_MAX,     "sat_add(max,max) saturates");
        expect(sat_mul(3, 4) == 12,                      "sat_mul(3,4)");
        expect(sat_mul(0, U64_MAX) == 0,                 "sat_mul(0,max)");
        expect(sat_mul(U64_MAX, 2) == U64_MAX,           "sat_mul saturates");
        expect(sat_mul(1ull << 33, 1ull << 33) == U64_MAX, "sat_mul(2^33,2^33) saturates");

        expect(sat_u64_from_double(-1.0) == 0,           "sat_u64_from_double(-1)");
        expect(sat_u64_from_double(0.0) == 0,            "sat_u64_from_double(0)");
        expect(sat_u64_from_double(1.5) == 1,            "sat_u64_from_double(1.5)");
        expect(sat_u64_from_double(1e40) == U64_MAX,     "sat_u64_from_double(1e40) saturates");
        expect(sat_u64_from_double(std::numeric_limits<double>::infinity()) == U64_MAX,
               "sat_u64_from_double(inf) saturates");
        expect(sat_u64_from_double(std::numeric_limits<double>::quiet_NaN()) == U64_MAX,
               "sat_u64_from_double(nan) saturates (strict direction)");

        expect(margin_mib_to_bytes(256) == 256ull * 1024 * 1024, "margin_mib_to_bytes(256)");
        expect(margin_mib_to_bytes(U64_MAX) == U64_MAX,          "margin saturates at max");
        expect(margin_mib_to_bytes(U64_MAX / (1024 * 1024) + 1) == U64_MAX,
               "margin saturates just past the clamp");
    }

    // ── JSON escaping: what @qvac/model-fit will JSON.parse ────────────────
    {
        expect(json_escape("plain") == "plain",              "json_escape passthrough");
        expect(json_escape("a\"b") == "a\\\"b",              "json_escape quote");
        expect(json_escape("a\\b") == "a\\\\b",              "json_escape backslash");
        expect(json_escape("a\nb\tc") == "a\\nb\\tc",        "json_escape newline/tab");
        expect(json_escape(std::string("a\x01") + "b") == "a\\u0001b",
               "json_escape control char");
        expect(json_escape("caf\xc3\xa9") == "caf\xc3\xa9",  "json_escape UTF-8 passthrough");
    }

    // ── CLI exit-code contract for garbage input (all fail before any model
    //    or backend work; the missing-model path is covered by the
    //    fixture-gated test_fit_params.cpp) ───────────────────────────────
    {
        const int err = (int) tts_cpp::acestep::FitStatus::Error;
        expect(run_cli({"--help"}) == 0,                            "--help exits 0");
        expect(run_cli({}) == err,                                  "missing models exits Error");
        expect(run_cli({"--bogus-flag"}) == err,                    "unknown flag exits Error");
        expect(run_cli({"--models-dir"}) == err,                    "--models-dir without value exits Error");
        expect(run_cli({"--models-dir", "x", "--duration", "abc"}) == err,
               "--duration garbage exits Error");
        expect(run_cli({"--models-dir", "x", "--duration", "0"}) == err,
               "--duration 0 exits Error");
        expect(run_cli({"--models-dir", "x", "--n-gpu-layers", "1x"}) == err,
               "--n-gpu-layers trailing garbage exits Error");
        expect(run_cli({"--models-dir", "x", "--threads", "two"}) == err,
               "--threads garbage exits Error");
        expect(run_cli({"--models-dir", "x", "--margin-mib", "-1"}) == err,
               "--margin-mib negative exits Error");
        expect(run_cli({"--models-dir", "x", "--text-tokens", "1.5"}) == err,
               "--text-tokens non-integer exits Error");
        expect(run_cli({"--models-dir", "x", "--lyric-tokens", "x"}) == err,
               "--lyric-tokens garbage exits Error");
        expect(run_cli({"--models-dir", "x", "--lm-cfg", "-2"}) == err,
               "--lm-cfg negative exits Error");
        expect(run_cli({"--models-dir", "x", "--guidance", "no"}) == err,
               "--guidance garbage exits Error");
        expect(run_cli({"--models-dir", "x", "--keep-stages", "2"}) == err,
               "--keep-stages out-of-range exits Error");
        expect(run_cli({"--models-dir", "x", "--lm-max-new-tokens", "many"}) == err,
               "--lm-max-new-tokens garbage exits Error");
        expect(run_cli({"--models-dir", "x", "--lm-prompt-tokens", "1e3"}) == err,
               "--lm-prompt-tokens non-integer exits Error");
        expect(run_cli({"--dit", "x.gguf"}) == err,
               "explicit paths missing a stage exit Error");
    }

    // ── Library-level argument validation (no models touched) ──────────────
    {
        tts_cpp::acestep::FitOptions o;  // no paths at all
        const tts_cpp::acestep::FitResult res = tts_cpp::acestep::fit_params(o);
        expect(res.status == tts_cpp::acestep::FitStatus::Error,
               "fit_params without models was not Error");
        expect(res.reason == "invalid-arguments",
               "fit_params without models reason was '" + res.reason + "'");
        expect(!res.fits, "fit_params without models reported fits");
    }

    if (g_failures == 0) {
        std::printf("test-fit-cli: all checks passed\n");
    }
    return g_failures;
}
