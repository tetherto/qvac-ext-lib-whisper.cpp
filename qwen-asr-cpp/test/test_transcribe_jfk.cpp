#include "qwen/engine.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string read_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string normalise(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char u = static_cast<unsigned char>(c);
        if (std::isalnum(u)) {
            out.push_back(static_cast<char>(std::tolower(u)));
        } else if (std::isspace(u)) {
            if (!out.empty() && out.back() != ' ') {
                out.push_back(' ');
            }
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

int fail(const char * msg) {
    std::fprintf(stderr, "test-transcribe-jfk: FAIL -- %s\n", msg);
    return 1;
}

}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: test-transcribe-jfk <model_dir> <wav> <ground_truth_txt>\n");
        return 2;
    }
    const std::string model_dir   = argv[1];
    const std::string wav_path    = argv[2];
    const std::string ground_path = argv[3];

    qwen::EngineOptions opts;
    opts.model_dir = model_dir;
    opts.language  = "English";
    opts.verbose   = 1;

    try {
        qwen::Engine engine(opts);
        auto result = engine.transcribe(wav_path);

        const std::string hyp = normalise(result.text);
        const std::string ref = normalise(read_file(ground_path));

        std::printf("ref: %s\n", ref.c_str());
        std::printf("hyp: %s\n", hyp.c_str());

        if (hyp.empty()) {
            return fail("hypothesis is empty");
        }
        if (hyp.find("ask not") == std::string::npos) {
            return fail("hypothesis does not contain the canonical 'ask not' phrase from jfk.wav");
        }
        if (hyp.find("country") == std::string::npos) {
            return fail("hypothesis does not contain 'country' from jfk.wav");
        }
        std::puts("test-transcribe-jfk: OK");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-transcribe-jfk: exception: %s\n", e.what());
        return 1;
    }
}
