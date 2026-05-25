#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen::gguf {

class Tokenizer {
public:
    Tokenizer(const std::vector<std::string> & vocab,
              const std::vector<std::string> & merges);

    std::vector<int32_t> encode(const std::string & text,
                                bool allow_specials = true) const;

    std::string decode(const std::vector<int32_t> & tokens,
                       bool skip_specials = false) const;

    std::string id_to_piece(int32_t id) const;

    int32_t piece_to_id(const std::string & piece) const;

    bool is_special(int32_t id) const;

    int32_t bos_id() const { return bos_; }
    int32_t eos_id() const { return eos_; }
    void set_bos(int32_t id) { bos_ = id; }
    void set_eos(int32_t id) { eos_ = id; }

private:
    void build_byte_unicode_maps();
    std::vector<std::string> pretokenize(const std::string & text, bool allow_specials) const;
    std::vector<int32_t> bpe_encode_word(const std::string & word) const;

    std::vector<std::string>                     id_to_piece_;
    std::unordered_map<std::string, int32_t>     piece_to_id_;
    std::unordered_map<uint64_t, int32_t>        merge_rank_;
    std::vector<std::string>                     specials_pieces_;
    std::vector<int32_t>                         specials_ids_;
    int                                          byte_to_unicode_[256] = {0};
    std::unordered_map<int, uint8_t>             unicode_to_byte_;
    int32_t                                      bos_ = -1;
    int32_t                                      eos_ = -1;
};

}
