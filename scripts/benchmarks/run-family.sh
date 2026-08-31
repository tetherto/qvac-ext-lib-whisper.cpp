#!/usr/bin/env bash
# Driver invoked by .github/workflows/benchmark-desktop.yml for one
# (family, runner) matrix cell. Also runnable locally:
#
#   scripts/benchmarks/run-family.sh \
#       --family whisper --runs 3 --build-dir build \
#       --models-root $PWD/bench-models --out result.json
#
# Contract: emits one JSON file (--out) with the shape
#
#   {
#     "family":         "whisper",
#     "model":          "whisper-tiny",       # or families.json[family].models[0]
#     "runner":         "ubuntu-24.04",       # $RUNNER_OS / matrix.os
#     "os":             "Linux",              # uname -s
#     "wall_ms_median": 1234.5,               # median across --runs iterations
#     "wall_ms_min":    1210.0,
#     "wall_ms_max":    1301.2,
#     "runs":           3,
#     "status":         "ok" | "missing-model" | "build-failed" | "run-failed",
#     "notes":          "..."
#   }
#
# The summarizer (scripts/benchmarks/summarize.py) reads these and produces
# the single markdown table for $GITHUB_STEP_SUMMARY.

set -euo pipefail

FAMILY=""
RUNS=3
WARMUP=1
BUILD_DIR="build"
MODELS_ROOT="$PWD/bench-models"
OUT="result.json"
RUNNER_LABEL="${RUNNER_OS:-unknown}"
AUDIO_DIR="engines/parakeet/test/samples"
WHISPER_SIZE="tiny"

usage() {
  cat <<EOF >&2
usage: $0 --family FAMILY [--runs N] [--warmup N]
          [--build-dir DIR] [--models-root DIR] [--audio-dir DIR]
          [--runner LABEL] [--whisper-size tiny|base|small]
          --out result.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family)         FAMILY="$2"; shift 2 ;;
    --runs)           RUNS="$2"; shift 2 ;;
    --warmup)         WARMUP="$2"; shift 2 ;;
    --build-dir)      BUILD_DIR="$2"; shift 2 ;;
    --models-root)    MODELS_ROOT="$2"; shift 2 ;;
    --audio-dir)      AUDIO_DIR="$2"; shift 2 ;;
    --runner)         RUNNER_LABEL="$2"; shift 2 ;;
    --whisper-size)   WHISPER_SIZE="$2"; shift 2 ;;
    --out)            OUT="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$FAMILY" ]]; then usage; exit 2; fi

# ---- read families.json -----------------------------------------------------
FAMILIES_JSON="$(dirname "$0")/families.json"
if ! [[ -f "$FAMILIES_JSON" ]]; then
  echo "families.json not found at $FAMILIES_JSON" >&2; exit 1
fi
if ! command -v jq >/dev/null; then
  echo "jq is required" >&2; exit 1
fi

spec_field() {
  jq -r --arg family "$FAMILY" --arg field "$1" '.[$family][$field] // ""' "$FAMILIES_JSON"
}

BENCH_KIND="$(spec_field bench_kind)"
BINARY_REL="$(spec_field binary)"
S3_PREFIX="$(spec_field s3_prefix)"
ARGS_TEMPLATE="$(spec_field args)"
NOTES="$(spec_field notes)"

if [[ -z "$BENCH_KIND" ]]; then
  echo "family '$FAMILY' not found in families.json" >&2; exit 1
fi

MODEL_DIR="$MODELS_ROOT/$FAMILY"
mkdir -p "$MODEL_DIR"

# ---- fetch models from S3 (skipped when models[] is empty / no bucket) ------
fetch_models() {
  local bucket="${MODEL_S3_BUCKET:-}"
  if [[ -z "$bucket" ]]; then
    echo "MODEL_S3_BUCKET not set — skipping fetch (assuming local models present)" >&2
    return 0
  fi

  local -a keys=()
  if [[ "$FAMILY" == "whisper" ]]; then
    mapfile -t keys < <(jq -r --arg size "$WHISPER_SIZE" \
      '.whisper.models_by_size[$size][]?' "$FAMILIES_JSON")
    if [[ ${#keys[@]} -eq 0 ]]; then
      echo "$FAMILY-$WHISPER_SIZE: no S3 key in registry (not-in-registry)"
      return 66  # sentinel: model not in registry
    fi
  else
    mapfile -t keys < <(jq -r --arg family "$FAMILY" \
      '.[$family].models[]?' "$FAMILIES_JSON")
    if [[ ${#keys[@]} -eq 0 ]]; then
      # minimax-shaped: no S3 path at all
      return 65
    fi
  fi

  for key in "${keys[@]}"; do
    local basename="${key##*/}"
    local dest="$MODEL_DIR/$basename"
    if [[ -f "$dest" ]]; then continue; fi
    local s3url="s3://$bucket/qvac_models_compiled/ggml/$S3_PREFIX/$key"
    echo "fetch $s3url -> $dest"
    aws s3 cp "$s3url" "$dest" --no-progress
  done
  return 0
}

# ---- emit-json helpers ------------------------------------------------------
emit_json() {
  local status="$1" median="${2:-null}" min="${3:-null}" max="${4:-null}" extra="${5:-}"
  local uname_s; uname_s="$(uname -s)"
  jq -n \
    --arg family "$FAMILY" \
    --arg model  "$MODEL_LABEL" \
    --arg runner "$RUNNER_LABEL" \
    --arg os     "$uname_s" \
    --argjson wall_median "$median" \
    --argjson wall_min "$min" \
    --argjson wall_max "$max" \
    --argjson runs "$RUNS" \
    --arg status "$status" \
    --arg notes  "$NOTES$extra" \
    '{family:$family, model:$model, runner:$runner, os:$os,
      wall_ms_median:$wall_median, wall_ms_min:$wall_min, wall_ms_max:$wall_max,
      runs:$runs, status:$status, notes:$notes}' > "$OUT"
  echo "wrote $OUT"
  cat "$OUT" >&2
}

# ---- resolve MODEL_LABEL + placeholders -------------------------------------
if [[ "$FAMILY" == "whisper" ]]; then
  MODEL_LABEL="whisper-$WHISPER_SIZE"
else
  first_model="$(jq -r --arg f "$FAMILY" '.[$f].models[0] // ""' "$FAMILIES_JSON")"
  MODEL_LABEL="$FAMILY: ${first_model##*/}"
fi

fetch_status=0
fetch_models || fetch_status=$?

if [[ $fetch_status -eq 66 ]]; then
  emit_json "not-in-registry" null null null " (whisper $WHISPER_SIZE size not in the model registry)"
  exit 0
fi
if [[ $fetch_status -eq 65 ]]; then
  emit_json "missing-model" null null null " (no S3 path in the registry — follow-up ticket)"
  exit 0
fi
if [[ $fetch_status -ne 0 ]]; then
  emit_json "run-failed" null null null " (model fetch failed with status $fetch_status)"
  exit 0
fi

# ---- pick the model path for whisper (single -m arg) ------------------------
if [[ "$FAMILY" == "whisper" ]]; then
  wkey="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size][0]' "$FAMILIES_JSON")"
  MODEL_PATH="$MODEL_DIR/${wkey##*/}"
fi

# ---- expand args placeholders -----------------------------------------------
EXPANDED_ARGS="${ARGS_TEMPLATE//\$\{MODEL_DIR\}/$MODEL_DIR}"
EXPANDED_ARGS="${EXPANDED_ARGS//\$\{MODELS_ROOT\}/$MODELS_ROOT}"
EXPANDED_ARGS="${EXPANDED_ARGS//\$\{AUDIO_DIR\}/$AUDIO_DIR}"
EXPANDED_ARGS="${EXPANDED_ARGS//\$\{MODEL_PATH\}/${MODEL_PATH:-}}"

BINARY="$BUILD_DIR/$BINARY_REL"
if ! [[ -x "$BINARY" ]]; then
  echo "binary not found or not executable: $BINARY" >&2
  emit_json "build-failed" null null null " (binary '$BINARY' missing)"
  exit 0
fi

# ---- run --------------------------------------------------------------------
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

run_one() {
  local iter="$1"
  local logfile="$tmp_dir/run-$iter.log"
  local start_ns end_ns
  start_ns="$(date +%s%N)"
  # shellcheck disable=SC2086  # intentional word-splitting on EXPANDED_ARGS
  if ! "$BINARY" $EXPANDED_ARGS > "$logfile" 2>&1; then
    echo "run $iter failed; log tail:" >&2
    tail -20 "$logfile" >&2
    return 1
  fi
  end_ns="$(date +%s%N)"
  # ms as a float with one decimal
  awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.1f", (e - s) / 1000000.0 }'
}

# Warmup
for i in $(seq 1 "$WARMUP"); do
  echo "warmup $i/$WARMUP" >&2
  if ! run_one "warmup-$i" >/dev/null; then
    emit_json "run-failed" null null null " (warmup failed)"; exit 0
  fi
done

# Timed
declare -a wall_ms=()
for i in $(seq 1 "$RUNS"); do
  echo "run $i/$RUNS" >&2
  if ! ms="$(run_one "$i")"; then
    emit_json "run-failed" null null null " (timed run $i failed)"; exit 0
  fi
  wall_ms+=("$ms")
done

# median / min / max via jq
stats="$(printf '%s\n' "${wall_ms[@]}" | jq -s '{
  median: (sort | if length%2==1 then .[length/2|floor] else (.[length/2-1] + .[length/2]) / 2 end),
  min: min,
  max: max
}')"
median="$(echo "$stats" | jq '.median')"
mn="$(echo "$stats" | jq '.min')"
mx="$(echo "$stats" | jq '.max')"

emit_json "ok" "$median" "$mn" "$mx"
