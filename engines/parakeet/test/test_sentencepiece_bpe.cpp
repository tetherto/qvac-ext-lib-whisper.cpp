#include "sentencepiece_bpe.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr const char * k_marker = "\xE2\x96\x81";

std::string marked(const char * piece) {
    return std::string(k_marker) + piece;
}

parakeet::BpeVocab make_vocab() {
    parakeet::BpeVocab vocab;
    vocab.pieces = {
        marked("see"),      // 0
        marked("if"),       // 1
        "ctuation",         // 2
        marked("pun"),      // 3
        ",",                // 4
        "<blank>",          // 5
        "<s>",              // 6
        "</s>",             // 7
        "<pad>",            // 8
        marked(""),         // 9: bare word-boundary marker
        "a" + marked("b"),  // 10: marker mid-piece
        "..",               // 11: two bytes, shorter than the marker
    };
    vocab.blank_id = 5;
    vocab.bos_id   = 6;
    vocab.eos_id   = 7;
    vocab.pad_id   = 8;
    return vocab;
}

bool expect_text(const parakeet::BpeVocab & vocab,
                 const std::vector<int32_t> & token_ids,
                 const std::string & expected, const char * scenario) {
    const std::string got = parakeet::detokenize(vocab, token_ids);
    if (got == expected) return true;
    std::fprintf(stderr, "%s: got \"%s\", expected \"%s\"\n", scenario,
                 got.c_str(), expected.c_str());
    return false;
}

bool check_words_join_with_spaces() {
    return expect_text(make_vocab(), {0, 1}, "see if", "words join");
}

bool check_continuation_glues_onto_previous_word() {
    return expect_text(make_vocab(), {3, 2}, "punctuation",
                       "continuation glues");
}

bool check_punctuation_glues_without_space() {
    return expect_text(make_vocab(), {0, 4, 1}, "see, if",
                       "punctuation glues");
}

bool check_leading_space_is_stripped() {
    return expect_text(make_vocab(), {0}, "see", "leading space stripped");
}

bool check_specials_are_skipped() {
    return expect_text(make_vocab(), {6, 5, 0, 8, 1, 7}, "see if",
                       "specials skipped");
}

bool check_out_of_range_ids_are_skipped() {
    const auto vocab = make_vocab();
    return expect_text(vocab, {-1, 0, (int32_t) vocab.pieces.size(), 1},
                       "see if", "out-of-range skipped");
}

bool check_marker_mid_piece_becomes_space() {
    return expect_text(make_vocab(), {0, 10}, "seea b", "mid-piece marker");
}

bool check_empty_input_yields_empty_text() {
    return expect_text(make_vocab(), {}, "", "empty input");
}

bool check_word_start_for_marked_pieces() {
    const auto vocab = make_vocab();
    return parakeet::token_is_word_start(vocab, 0) &&
           parakeet::token_is_word_start(vocab, 1) &&
           parakeet::token_is_word_start(vocab, 3) &&
           parakeet::token_is_word_start(vocab, 9);
}

bool check_word_start_false_for_continuations_and_punctuation() {
    const auto vocab = make_vocab();
    return !parakeet::token_is_word_start(vocab, 2) &&
           !parakeet::token_is_word_start(vocab, 4) &&
           !parakeet::token_is_word_start(vocab, 10) &&
           !parakeet::token_is_word_start(vocab, 11);
}

bool check_word_start_false_for_specials_and_out_of_range() {
    const auto vocab = make_vocab();
    return !parakeet::token_is_word_start(vocab, vocab.blank_id) &&
           !parakeet::token_is_word_start(vocab, vocab.bos_id) &&
           !parakeet::token_is_word_start(vocab, vocab.eos_id) &&
           !parakeet::token_is_word_start(vocab, vocab.pad_id) &&
           !parakeet::token_is_word_start(vocab, -1) &&
           !parakeet::token_is_word_start(vocab,
                                          (int32_t) vocab.pieces.size());
}

}

int main() {
    int failures = 0;

    failures += check_words_join_with_spaces() ? 0 : 1;
    failures += check_continuation_glues_onto_previous_word() ? 0 : 1;
    failures += check_punctuation_glues_without_space() ? 0 : 1;
    failures += check_leading_space_is_stripped() ? 0 : 1;
    failures += check_specials_are_skipped() ? 0 : 1;
    failures += check_out_of_range_ids_are_skipped() ? 0 : 1;
    failures += check_marker_mid_piece_becomes_space() ? 0 : 1;
    failures += check_empty_input_yields_empty_text() ? 0 : 1;
    failures += check_word_start_for_marked_pieces() ? 0 : 1;
    failures += check_word_start_false_for_continuations_and_punctuation()
                    ? 0 : 1;
    failures += check_word_start_false_for_specials_and_out_of_range() ? 0 : 1;

    if (failures > 0) {
        std::fprintf(stderr, "test_sentencepiece_bpe: %d failure(s)\n",
                     failures);
        return 1;
    }
    std::printf("test_sentencepiece_bpe: all checks passed\n");
    return 0;
}
