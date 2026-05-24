#!/usr/bin/env bash
# Download Qwen3-ASR model checkpoints from HuggingFace into ./models/hf/.
# Idempotent: skips files already present on disk.
#
# Usage:
#   ./scripts/download-models.sh [flags]
#
# Flags:
#   --type, -t <0.6b|1.7b|all>    Which model(s) to download (default: 0.6b)
#   --output, -o <path>           Destination root dir (default: ./models/hf)
#   --force, -f                   Re-download even if present
#   --help, -h                    Show this help

set -euo pipefail

TYPE="0.6b"
OUTPUT_DIR="./models/hf"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type|-t)   TYPE="$2"; shift 2;;
    --output|-o) OUTPUT_DIR="$2"; shift 2;;
    --force|-f)  FORCE=1; shift;;
    --help|-h)
      sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed -e '/^set -euo/d' -e 's/^# *//' >&2
      exit 0;;
    *) echo "Unknown flag: $1" >&2; exit 2;;
  esac
done

case "$TYPE" in
  0.6b|1.7b|all) ;;
  *) echo "Error: --type must be 0.6b|1.7b|all" >&2; exit 2;;
esac

hf_repo() {
  case "$1" in
    0.6b) echo "Qwen/Qwen3-ASR-0.6B";;
    1.7b) echo "Qwen/Qwen3-ASR-1.7B";;
  esac
}

REQUIRED_FILES=(
  config.json
  generation_config.json
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors
)

fetch_one() {
  local t="$1"
  local repo; repo="$(hf_repo "$t")"
  local dst="${OUTPUT_DIR}/${t}"
  mkdir -p "${dst}"

  for f in "${REQUIRED_FILES[@]}"; do
    local out="${dst}/${f}"
    if [[ -s "$out" ]] && [[ "$FORCE" -eq 0 ]]; then
      echo "  - ${t}/${f}: already present"
      continue
    fi
    local url="https://huggingface.co/${repo}/resolve/main/${f}"
    echo "  v ${t}/${f}: ${url}"
    curl -L --fail --progress-bar -o "${out}.tmp" "${url}"
    mv "${out}.tmp" "${out}"
  done
}

echo "Downloading Qwen3-ASR checkpoint(s) -- type=${TYPE}"
echo "Output: ${OUTPUT_DIR}"
echo

if [[ "$TYPE" == "all" ]]; then
  for t in 0.6b 1.7b; do
    fetch_one "$t"
  done
else
  fetch_one "$TYPE"
fi

echo
echo "Next: build and transcribe"
echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"
echo "  ./build/qwen-asr transcribe --model-dir ${OUTPUT_DIR}/${TYPE} --wav test/samples/jfk.wav --language English"
