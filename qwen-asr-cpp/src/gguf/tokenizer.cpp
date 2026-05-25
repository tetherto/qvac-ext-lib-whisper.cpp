#include "tokenizer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace qwen::gguf {

namespace {

void utf8_emit_cp(int cp, std::string & out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
        return;
    }
    if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        return;
    }
    if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        return;
    }
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
}

int utf8_take_cp(const std::string & s, size_t & i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    int cp = c;
    int n  = 1;
    if      ((c & 0x80) == 0)    { n = 1; cp = c; }
    else if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07; }
    for (int k = 1; k < n && i + k < s.size(); ++k) {
        cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    }
    i += n;
    return cp;
}

bool is_special_piece(const std::string & p) {
    return p.size() >= 4 && p.front() == '<' && p[1] == '|' &&
           p[p.size() - 2] == '|' && p.back() == '>';
}

uint64_t pair_key(int32_t a, int32_t b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(b));
}

}

Tokenizer::Tokenizer(const std::vector<std::string> & vocab,
                     const std::vector<std::string> & merges) {
    build_byte_unicode_maps();

    id_to_piece_.assign(vocab.begin(), vocab.end());
    piece_to_id_.reserve(vocab.size() * 2);
    for (size_t i = 0; i < vocab.size(); ++i) {
        piece_to_id_[vocab[i]] = static_cast<int32_t>(i);
        if (is_special_piece(vocab[i])) {
            specials_pieces_.push_back(vocab[i]);
            specials_ids_.push_back(static_cast<int32_t>(i));
        }
    }

    merge_rank_.reserve(merges.size() * 2);
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string & line = merges[rank];
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        const std::string a = line.substr(0, sp);
        const std::string b = line.substr(sp + 1);
        const auto ia = piece_to_id_.find(a);
        const auto ib = piece_to_id_.find(b);
        if (ia == piece_to_id_.end() || ib == piece_to_id_.end()) continue;
        merge_rank_.emplace(pair_key(ia->second, ib->second), static_cast<int32_t>(rank));
    }
}

void Tokenizer::build_byte_unicode_maps() {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool is_normal =
            (b >= 33 && b <= 126) ||
            (b >= 161 && b <= 172) ||
            (b >= 174 && b <= 255);
        const int cp = is_normal ? b : (256 + n++);
        byte_to_unicode_[b]    = cp;
        unicode_to_byte_[cp]   = static_cast<uint8_t>(b);
    }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string & text, bool allow_specials) const {
    std::vector<std::string> chunks;
    if (!allow_specials || specials_pieces_.empty()) {
        chunks.push_back(text);
        return chunks;
    }
    size_t i = 0;
    while (i < text.size()) {
        size_t best_pos = std::string::npos;
        size_t best_len = 0;
        const std::string * best_match = nullptr;
        for (const auto & sp : specials_pieces_) {
            const size_t pos = text.find(sp, i);
            if (pos != std::string::npos && (pos < best_pos || best_match == nullptr)) {
                best_pos   = pos;
                best_len   = sp.size();
                best_match = &sp;
            }
        }
        if (best_match == nullptr) {
            chunks.push_back(text.substr(i));
            break;
        }
        if (best_pos > i) {
            chunks.push_back(text.substr(i, best_pos - i));
        }
        chunks.push_back(*best_match);
        i = best_pos + best_len;
    }
    return chunks;
}

std::vector<int32_t> Tokenizer::bpe_encode_word(const std::string & word) const {
    std::vector<int32_t> ids;
    if (word.empty()) return ids;
    std::vector<int32_t> symbols;
    symbols.reserve(word.size() * 2);
    for (size_t i = 0; i < word.size(); ) {
        const int cp = utf8_take_cp(word, i);
        std::string piece;
        utf8_emit_cp(cp, piece);
        const auto it = piece_to_id_.find(piece);
        if (it != piece_to_id_.end()) {
            symbols.push_back(it->second);
        } else {
            return ids;
        }
    }
    while (true) {
        int best_pos  = -1;
        int best_rank = std::numeric_limits<int>::max();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = merge_rank_.find(pair_key(symbols[i], symbols[i + 1]));
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = static_cast<int>(i);
            }
        }
        if (best_pos < 0) break;
        const std::string merged =
            id_to_piece_[symbols[best_pos]] + id_to_piece_[symbols[best_pos + 1]];
        const auto it = piece_to_id_.find(merged);
        if (it == piece_to_id_.end()) break;
        symbols[best_pos] = it->second;
        symbols.erase(symbols.begin() + best_pos + 1);
    }
    ids.insert(ids.end(), symbols.begin(), symbols.end());
    return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool allow_specials) const {
    std::vector<int32_t> ids;
    for (const auto & chunk : pretokenize(text, allow_specials)) {
        if (allow_specials) {
            const auto sp = piece_to_id_.find(chunk);
            if (sp != piece_to_id_.end() && is_special_piece(chunk)) {
                ids.push_back(sp->second);
                continue;
            }
        }
        std::string mapped;
        mapped.reserve(chunk.size() * 2);
        for (unsigned char c : chunk) {
            const int cp = byte_to_unicode_[c];
            utf8_emit_cp(cp, mapped);
        }
        const auto sub = bpe_encode_word(mapped);
        ids.insert(ids.end(), sub.begin(), sub.end());
    }
    return ids;
}

std::string Tokenizer::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_piece_.size()) return "";
    return id_to_piece_[id];
}

int32_t Tokenizer::piece_to_id(const std::string & piece) const {
    const auto it = piece_to_id_.find(piece);
    return (it == piece_to_id_.end()) ? -1 : it->second;
}

bool Tokenizer::is_special(int32_t id) const {
    return std::find(specials_ids_.begin(), specials_ids_.end(), id) != specials_ids_.end();
}

std::string Tokenizer::decode(const std::vector<int32_t> & tokens, bool skip_specials) const {
    std::string out;
    std::string buf;
    for (int32_t id : tokens) {
        if (skip_specials && is_special(id)) continue;
        const std::string & piece = id_to_piece(id);
        if (is_special_piece(piece)) {
            if (!skip_specials) out.append(piece);
            continue;
        }
        for (size_t i = 0; i < piece.size(); ) {
            const int cp = utf8_take_cp(piece, i);
            const auto it = unicode_to_byte_.find(cp);
            if (it != unicode_to_byte_.end()) {
                out.push_back(static_cast<char>(it->second));
            } else {
                utf8_emit_cp(cp, out);
            }
        }
    }
    return out;
}

}
