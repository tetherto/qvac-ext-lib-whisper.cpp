#pragma once

#include "hparams.h"

#include <ggml.h>
#include <gguf.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen::gguf {

struct LoadedTensor {
    ggml_tensor * tensor = nullptr;
    size_t        offset = 0;
    size_t        nbytes = 0;
};

class ModelLoader {
public:
    explicit ModelLoader(const std::string & path);
    ~ModelLoader();

    ModelLoader(const ModelLoader &)             = delete;
    ModelLoader & operator=(const ModelLoader &) = delete;

    const Hparams & hparams() const { return hparams_; }

    const std::vector<std::string> & vocab() const { return vocab_; }
    const std::vector<std::string> & merges() const { return merges_; }

    const std::unordered_map<std::string, LoadedTensor> & tensors() const { return tensors_; }

    const LoadedTensor * find(const std::string & name) const;

    bool has(const std::string & name) const { return find(name) != nullptr; }

    ggml_tensor * tensor(const std::string & name) const;

    void read_into(const LoadedTensor & lt, void * dst) const;

    std::vector<std::string> missing(const std::vector<std::string> & required) const;

    size_t total_bytes() const { return total_bytes_; }

    ggml_context * weights_ctx() const { return ctx_; }

private:
    void load_metadata();
    void load_vocab();
    void load_tensor_table();
    void map_weights();

    std::string         path_;
    gguf_context *      gctx_ = nullptr;
    ggml_context *      ctx_  = nullptr;
    Hparams             hparams_;
    std::vector<std::string> vocab_;
    std::vector<std::string> merges_;
    std::unordered_map<std::string, LoadedTensor> tensors_;
    size_t              data_start_ = 0;
    size_t              total_bytes_ = 0;
    void *              file_mmap_  = nullptr;
    size_t              file_size_  = 0;
    int                 file_fd_    = -1;
};

std::vector<std::string> expected_tensor_names(const Hparams & h);

}
