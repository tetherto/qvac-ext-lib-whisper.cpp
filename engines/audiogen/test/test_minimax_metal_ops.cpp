#include "minimax/mm3-cond-graph.h"
#include "minimax/mm3-vocoder-graph.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t INPUT_LENGTH = 17;
constexpr int64_t INPUT_CHANNELS = 8;
constexpr int64_t OUTPUT_CHANNELS = 4;
constexpr int64_t CONDITION_CHANNELS = 4096;
constexpr int64_t CONDITION_OUTPUT_CHANNELS = 4;
constexpr int64_t CONDITION_FRAMES = 17;
constexpr float ABSOLUTE_TOLERANCE = 1e-4f;

std::vector<float> make_values(size_t count, float frequency, float amplitude = 1.0f) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = amplitude * std::sin(static_cast<float>(i + 1) * frequency);
    }
    return values;
}

ggml_backend_t find_metal_backend() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        const char * name = registry ? ggml_backend_reg_name(registry) : nullptr;
        if (name && (std::strcmp(name, "MTL") == 0 || std::strcmp(name, "Metal") == 0)) {
            return ggml_backend_dev_init(device, nullptr);
        }
    }
    return nullptr;
}

std::vector<float> run_condition_projection(ggml_backend_t backend) {
    constexpr size_t MAX_NODES = 16;
    const size_t context_size =
        ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    ggml_init_params params = {context_size, nullptr, true};
    ggml_context * context = ggml_init(params);
    ggml_cgraph * graph = ggml_new_graph_custom(context, MAX_NODES, false);

    ggml_tensor * weights = ggml_new_tensor_3d(
        context, GGML_TYPE_F32, 3, CONDITION_CHANNELS, CONDITION_OUTPUT_CHANNELS);
    ggml_tensor * input =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, CONDITION_FRAMES, CONDITION_CHANNELS);
    ggml_set_input(weights);
    ggml_set_input(input);
    ggml_tensor * output = mm3_cond_conv1d(context, weights, nullptr, input, 1);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocator, graph);
    ggml_gallocr_alloc_graph(allocator, graph);
    const std::vector<float> weight_values =
        make_values(static_cast<size_t>(ggml_nelements(weights)), 0.013f, 0.01f);
    const std::vector<float> input_values =
        make_values(static_cast<size_t>(ggml_nelements(input)), 0.031f, 0.1f);
    ggml_backend_tensor_set(weights, weight_values.data(), 0, ggml_nbytes(weights));
    ggml_backend_tensor_set(input, input_values.data(), 0, ggml_nbytes(input));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> values(static_cast<size_t>(ggml_nelements(output)));
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    } else {
        values.clear();
    }
    ggml_gallocr_free(allocator);
    ggml_free(context);
    return values;
}

bool condition_outputs_match(const std::vector<float> & expected,
                             const std::vector<float> & actual) {
    if (expected.size() != actual.size() || expected.empty()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float difference = std::fabs(expected[i] - actual[i]);
        if (!std::isfinite(actual[i]) || difference > 1e-3f) {
            std::fprintf(stderr,
                         "FAIL condition projection index=%zu cpu=%.8f metal=%.8f difference=%.8f\n",
                         i,
                         expected[i],
                         actual[i],
                         difference);
            return false;
        }
    }
    return true;
}

bool verify_condition_projection(ggml_backend_t cpu, ggml_backend_t metal) {
    const std::vector<float> expected = run_condition_projection(cpu);
    const std::vector<float> actual = run_condition_projection(metal);
    if (!condition_outputs_match(expected, actual)) return false;
    std::fprintf(stderr, "PASS MiniMax condition projection\n");
    return true;
}

std::vector<float> run_vocoder_convt(ggml_backend_t backend, int stride) {
    constexpr size_t MAX_NODES = 64;
    const size_t context_size =
        ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    ggml_init_params params = {context_size, nullptr, true};
    ggml_context * context = ggml_init(params);
    ggml_cgraph * graph = ggml_new_graph_custom(context, MAX_NODES, false);

    const int64_t kernel_size = 2 * stride;
    ggml_tensor * weights =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, INPUT_CHANNELS, kernel_size * OUTPUT_CHANNELS);
    ggml_tensor * input =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, INPUT_LENGTH, INPUT_CHANNELS);
    ggml_set_name(weights, "weights");
    ggml_set_name(input, "input");
    ggml_set_input(weights);
    ggml_set_input(input);

    ggml_tensor * output =
        mm3_voc_convt(context, weights, nullptr, input, stride, OUTPUT_CHANNELS);
    ggml_set_name(output, "output");
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocator, graph);
    ggml_gallocr_alloc_graph(allocator, graph);

    const std::vector<float> weight_values =
        make_values(static_cast<size_t>(ggml_nelements(weights)), 0.013f);
    const std::vector<float> input_values =
        make_values(static_cast<size_t>(ggml_nelements(input)), 0.031f);
    ggml_backend_tensor_set(weights, weight_values.data(), 0, ggml_nbytes(weights));
    ggml_backend_tensor_set(input, input_values.data(), 0, ggml_nbytes(input));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> values(static_cast<size_t>(ggml_nelements(output)));
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    } else {
        values.clear();
    }

    ggml_gallocr_free(allocator);
    ggml_free(context);
    return values;
}

bool outputs_match(const std::vector<float> & expected,
                   const std::vector<float> & actual,
                   int stride) {
    if (expected.size() != actual.size() || expected.empty()) {
        std::fprintf(stderr,
                     "FAIL stride=%d output size cpu=%zu metal=%zu\n",
                     stride,
                     expected.size(),
                     actual.size());
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        const float difference = std::fabs(expected[i] - actual[i]);
        if (!std::isfinite(actual[i]) || difference > ABSOLUTE_TOLERANCE) {
            std::fprintf(stderr,
                         "FAIL stride=%d index=%zu cpu=%.8f metal=%.8f difference=%.8f\n",
                         stride,
                         i,
                         expected[i],
                         actual[i],
                         difference);
            return false;
        }
    }
    return true;
}

bool verify_vocoder_strides(ggml_backend_t cpu, ggml_backend_t metal) {
    constexpr int STRIDES[] = {8, 4, 2};
    for (int stride : STRIDES) {
        const std::vector<float> expected = run_vocoder_convt(cpu, stride);
        const std::vector<float> actual = run_vocoder_convt(metal, stride);
        if (!outputs_match(expected, actual, stride)) return false;
        std::fprintf(stderr, "PASS MiniMax vocoder convt stride=%d\n", stride);
    }
    return true;
}

}

int main() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::fprintf(stderr, "FAIL CPU backend initialization failed\n");
        return 1;
    }
    ggml_backend_t metal = find_metal_backend();
    if (!metal) {
        ggml_backend_free(cpu);
        std::fprintf(stderr, "SKIP Metal backend is unavailable\n");
        return 77;
    }

    const bool passed =
        verify_condition_projection(cpu, metal) && verify_vocoder_strides(cpu, metal);
    ggml_backend_free(metal);
    ggml_backend_free(cpu);
    return passed ? 0 : 1;
}
