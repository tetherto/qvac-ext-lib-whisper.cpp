#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="${ROOT}/build/qwen-asr"
SAMPLES_DIR="${ROOT}/native-app/ios/QwenAsrDemo/samples"
MODEL_GGUF="${ROOT}/models/gguf/qwen3-asr-0.6b.gguf"
MODEL_SAFE="${ROOT}/models/hf/0.6b"
RESULTS_DIR="${ROOT}/build/test-langs-out"

BACKENDS_ARG="${1:-both}"

mkdir -p "${RESULTS_DIR}"

require_file() {
    local path="$1"
    local what="$2"
    if [[ ! -e "${path}" ]]; then
        echo "[test-all-langs] missing ${what}: ${path}" >&2
        exit 1
    fi
}

require_file "${CLI}" "qwen-asr CLI (build it first: cmake --build build)"
require_file "${SAMPLES_DIR}" "samples directory"

resolve_backends() {
    case "${BACKENDS_ARG}" in
        gguf)         echo "gguf" ;;
        safetensors)  echo "safetensors" ;;
        both|"")      echo "safetensors gguf" ;;
        *) echo "[test-all-langs] unknown backend: ${BACKENDS_ARG}" >&2; exit 2 ;;
    esac
}

model_for_backend() {
    case "$1" in
        safetensors) echo "${MODEL_SAFE}" ;;
        gguf)        echo "${MODEL_GGUF}" ;;
    esac
}

list_languages() {
    find "${SAMPLES_DIR}" -name '*.wav' -type f -print0 | xargs -0 -n1 basename | sed 's/\.wav$//' | sort
}

run_single() {
    local backend="$1"
    local lang="$2"
    local wav="${SAMPLES_DIR}/${lang}.wav"
    local txt="${SAMPLES_DIR}/${lang}.txt"
    local model
    model="$(model_for_backend "${backend}")"
    local out_file="${RESULTS_DIR}/${lang}-${backend}.out"
    local err_file="${RESULTS_DIR}/${lang}-${backend}.err"
    local rc=0
    local t_start t_end
    t_start="$(python3 -c 'import time; print(time.perf_counter())')"
    if "${CLI}" transcribe --backend "${backend}" --model "${model}" --wav "${wav}" \
            >"${out_file}" 2>"${err_file}"; then
        rc=0
    else
        rc=$?
    fi
    t_end="$(python3 -c 'import time; print(time.perf_counter())')"
    local elapsed
    elapsed="$(python3 -c "print(f'{(${t_end}-${t_start})*1000.0:.0f}')")"
    printf "%s|%s|%s|%s\n" "${lang}" "${backend}" "${rc}" "${elapsed}"
}

print_header() {
    printf "\n%-6s | %-12s | %-3s | %-7s | %s\n" "LANG" "BACKEND" "RC" "MS" "OUTPUT (first 80 chars)"
    printf "%s\n" "------ + ------------ + --- + ------- + -------------------------------------"
}

print_result() {
    local lang="$1" backend="$2" rc="$3" elapsed="$4"
    local out="${RESULTS_DIR}/${lang}-${backend}.out"
    local first_line preview
    if [[ -s "${out}" ]]; then
        first_line="$(head -1 "${out}")"
        preview="${first_line:0:80}"
    else
        preview="(empty output)"
    fi
    local status_icon="OK "
    if [[ "${rc}" != "0" ]]; then status_icon="ERR"; fi
    printf "%-6s | %-12s | %-3s | %-7s | %s\n" "${lang}" "${backend}" "${status_icon}" "${elapsed}" "${preview}"
}

main() {
    local backends
    backends="$(resolve_backends)"
    local languages
    languages="$(list_languages)"
    print_header
    local total=0 ok=0 err=0
    for lang in ${languages}; do
        for backend in ${backends}; do
            local raw
            raw="$(run_single "${backend}" "${lang}")"
            IFS='|' read -r r_lang r_backend r_rc r_elapsed <<< "${raw}"
            print_result "${r_lang}" "${r_backend}" "${r_rc}" "${r_elapsed}"
            total=$((total + 1))
            if [[ "${r_rc}" == "0" ]]; then ok=$((ok + 1)); else err=$((err + 1)); fi
        done
    done
    printf "\n[test-all-langs] %d runs total | %d OK | %d ERR\n" "${total}" "${ok}" "${err}"
    printf "[test-all-langs] outputs in: %s\n" "${RESULTS_DIR}"
    if [[ "${err}" -gt 0 ]]; then exit 1; fi
}

main "$@"
