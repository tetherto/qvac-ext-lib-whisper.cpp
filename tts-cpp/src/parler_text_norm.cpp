#include "parler_text_norm.h"

#include <cstddef>
#include <string>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

const char * kOnes[20] = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen"};
const char * kTens[10] = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety"};
const char * kScales[6] = {"", "thousand", "million", "billion", "trillion",
    "quadrillion"};

bool is_digit(char c)       { return c >= '0' && c <= '9'; }
bool is_ascii_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
char lower(char c)          { return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c; }

std::string three_digit_words(int v) { // 1..999
    std::string out;
    if (v >= 100) {
        out += kOnes[v / 100];
        out += " hundred";
        v %= 100;
        if (v) out += ' ';
    }
    if (v >= 20) {
        out += kTens[v / 10];
        if (v % 10) { out += '-'; out += kOnes[v % 10]; }
    } else if (v > 0) {
        out += kOnes[v];
    }
    return out;
}

std::string cardinal_words(unsigned long long v) {
    if (v == 0) return kOnes[0];
    int groups[6] = {0};
    int n = 0;
    while (v > 0) { groups[n++] = (int)(v % 1000); v /= 1000; }
    std::string out;
    for (int g = n - 1; g >= 0; --g) {
        if (!groups[g]) continue;
        if (!out.empty()) out += ' ';
        out += three_digit_words(groups[g]);
        if (g) { out += ' '; out += kScales[g]; }
    }
    return out;
}

std::string digit_by_digit(const std::string & digits) {
    std::string out;
    for (char c : digits) {
        if (!out.empty()) out += ' ';
        out += kOnes[c - '0'];
    }
    return out;
}

// "twenty-one" -> "twenty-first": ordinalize the final word.
std::string ordinal_words(std::string words) {
    const size_t cut = words.find_last_of(" -");
    const size_t at  = cut == std::string::npos ? 0 : cut + 1;
    const std::string last = words.substr(at);
    static const struct { const char * card; const char * ord; } kSpecial[] = {
        {"one", "first"}, {"two", "second"}, {"three", "third"},
        {"five", "fifth"}, {"eight", "eighth"}, {"nine", "ninth"},
        {"twelve", "twelfth"},
    };
    for (const auto & s : kSpecial) {
        if (last == s.card) {
            words.replace(at, last.size(), s.ord);
            return words;
        }
    }
    if (!last.empty() && last.back() == 'y') {
        words.replace(words.size() - 1, 1, "ieth"); // twenty -> twentieth
    } else {
        words += "th"; // four / six / hundred / thousand -> fourth / ...
    }
    return words;
}

} // namespace

std::string normalize_numbers_en(const std::string & text) {
    const size_t n = text.size();
    std::string out;
    out.reserve(n);

    size_t i = 0;
    while (i < n) {
        if (!is_digit(text[i])) {
            out += text[i++];
            continue;
        }

        size_t j = i;
        while (j < n && is_digit(text[j])) ++j;
        std::string ds = text.substr(i, j - i);

        // ",ddd" thousands separators: only after a 1-3 digit lead group,
        // and only for full groups not followed by a fourth digit
        if (ds.size() <= 3) {
            while (j + 4 <= n && text[j] == ',' &&
                   is_digit(text[j + 1]) && is_digit(text[j + 2]) &&
                   is_digit(text[j + 3]) &&
                   (j + 4 == n || !is_digit(text[j + 4]))) {
                ds.append(text, j + 1, 3);
                j += 4;
            }
        }

        std::string frac;
        if (j + 1 < n && text[j] == '.' && is_digit(text[j + 1])) {
            size_t k = j + 1;
            while (k < n && is_digit(text[k])) ++k;
            frac.assign(text, j + 1, k - j - 1);
            j = k;
        }

        bool ordinal = false, plural = false;
        if (frac.empty() && j + 1 < n) {
            const char a = lower(text[j]), b = lower(text[j + 1]);
            const bool sfx = (a == 's' && b == 't') || (a == 'n' && b == 'd') ||
                             (a == 'r' && b == 'd') || (a == 't' && b == 'h');
            if (sfx) {
                size_t k = j + 2;
                const bool pl = k < n && lower(text[k]) == 's';
                if (pl) ++k;
                if (k == n || (!is_ascii_alpha(text[k]) && !is_digit(text[k]))) {
                    ordinal = true;
                    plural  = pl;
                    j       = k;
                }
            }
        }

        // leading zeros and >15-digit runs read digit-by-digit
        const bool dbd = ds.size() > 15 || (ds.size() > 1 && ds[0] == '0');
        std::string words = dbd ? digit_by_digit(ds)
                                : cardinal_words(std::stoull(ds));
        if (!frac.empty()) {
            words += " point ";
            words += digit_by_digit(frac);
        } else if (ordinal) {
            words = ordinal_words(std::move(words));
            if (plural) words += 's';
        }

        // keep words separated from abutting letters (and any UTF-8 bytes)
        if (!out.empty() &&
            (is_ascii_alpha(out.back()) || (unsigned char) out.back() >= 0x80)) {
            out += ' ';
        }
        out += words;
        if (j < n &&
            (is_ascii_alpha(text[j]) || (unsigned char) text[j] >= 0x80)) {
            out += ' ';
        }
        i = j;
    }
    return out;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
