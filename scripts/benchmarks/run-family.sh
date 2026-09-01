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
#     "model":          "whisper-tiny",
#     "runner":         "linux",
#     "os":             "Linux",
#     "backend":        "CUDA" | "Metal" | "Vulkan" | "OpenCL" | "CPU" | "unknown",
#     "wall_ms_median": 1234.5,
#     "wall_ms_min":    1210.0,
#     "wall_ms_max":    1301.2,
#     "rtf_median":     0.612,        # null when the family's rtf can't be measured
#     "peak_rss_mib":   1024.5,       # null on runners without /usr/bin/time -v/-l
#     "runs":           3,
#     "status":         "ok" | "not-in-registry" | "missing-model"
#                       | "build-failed" | "run-failed" | "fetch-failed",
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
spec_field_raw() {
  # For fields that may be null (audio_duration_seconds) — return jq's raw
  # "null" instead of an empty string so downstream can distinguish.
  jq -r --arg family "$FAMILY" --arg field "$1" '.[$family][$field]' "$FAMILIES_JSON"
}

BENCH_KIND="$(spec_field bench_kind)"
BINARY_REL="$(spec_field binary)"
S3_PREFIX="$(spec_field s3_prefix)"
ARGS_TEMPLATE="$(spec_field args)"
NOTES="$(spec_field notes)"
AUDIO_DURATION_S="$(spec_field_raw audio_duration_seconds)"   # "null" or a number

if [[ -z "$BENCH_KIND" ]]; then
  echo "family '$FAMILY' not found in families.json" >&2; exit 1
fi

MODEL_DIR="$MODELS_ROOT/$FAMILY"
mkdir -p "$MODEL_DIR"

# ---- output-JSON emitter (used from every exit path) ------------------------
BACKEND="unknown"     # populated after the first successful run parses stderr
PEAK_RSS_MIB="null"   # tracked across runs; max seen
RTF_MEDIAN="null"     # from bench JSON (native) or computed (time-wrapped w/ audio_duration_seconds)

emit_json() {
  local status="$1" median="${2:-null}" wmin="${3:-null}" wmax="${4:-null}" extra="${5:-}"
  local uname_s; uname_s="$(uname -s)"
  jq -n \
    --arg  family "$FAMILY" \
    --arg  model  "$MODEL_LABEL" \
    --arg  runner "$RUNNER_LABEL" \
    --arg  os     "$uname_s" \
    --arg  backend "$BACKEND" \
    --argjson wall_median "$median" \
    --argjson wall_min    "$wmin" \
    --argjson wall_max    "$wmax" \
    --argjson rtf_median  "$RTF_MEDIAN" \
    --argjson peak_rss    "$PEAK_RSS_MIB" \
    --argjson runs        "$RUNS" \
    --arg  status "$status" \
    --arg  notes  "$NOTES$extra" \
    '{family:$family, model:$model, runner:$runner, os:$os, backend:$backend,
      wall_ms_median:$wall_median, wall_ms_min:$wall_min, wall_ms_max:$wall_max,
      rtf_median:$rtf_median, peak_rss_mib:$peak_rss,
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

# ---- fetch models from S3 (skipped when models[] empty / no bucket) --------
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
      return 66
    fi
  else
    mapfile -t keys < <(jq -r --arg family "$FAMILY" \
      '.[$family].models[]?' "$FAMILIES_JSON")
    if [[ ${#keys[@]} -eq 0 ]]; then
      return 65      # minimax shape: no S3 path
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

# ---- pick the model path for whisper (single -m arg) ------------------------
MODEL_PATH=""
if [[ "$FAMILY" == "whisper" ]]; then
  wkey="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size][0] // ""' "$FAMILIES_JSON")"
  if [[ -n "$wkey" ]]; then
    MODEL_PATH="$MODEL_DIR/${wkey##*/}"
  fi
fi

fetch_status=0
fetch_models || fetch_status=$?
if   [[ $fetch_status -eq 66 ]]; then
  emit_json "not-in-registry" null null null " (whisper $WHISPER_SIZE size not in the model registry)"; exit 0
elif [[ $fetch_status -eq 65 ]]; then
  emit_json "missing-model"   null null null " (no S3 path in the registry — follow-up ticket)"; exit 0
elif [[ $fetch_status -ne 0 ]]; then
  emit_json "fetch-failed"    null null null " (model fetch failed with status $fetch_status)"; exit 0
fi

# ---- expand args placeholders -----------------------------------------------
EXPANDED_ARGS="${ARGS_TEMPLATE//\$\{MODEL_DIR\}/$MODEL_DIR}"
EXPANDED_ARGS="${EXPANDED_ARGS//\$\{MODELS_ROOT\}/$MODELS_ROOT}"
EXPANDED_ARGS="${EXPANDED_ARGS//\$\{AUDIO_DIR\}/$AUDIO_DIR}"
EXPANDED_ARGS="${EXPANDED_ARGS//\$\{MODEL_PATH\}/${MODEL_PATH:-}}"

BINARY="$BUILD_DIR/$BINARY_REL"
if ! [[ -x "$BINARY" ]]; then
  echo "binary not found or not executable: $BINARY" >&2
  emit_json "build-failed" null null null " (binary '$BINARY' missing)"; exit 0
fi

# ---- /usr/bin/time wrapper (portable RSS capture) --------------------------
# Linux GNU time: --verbose prints 'Maximum resident set size (kbytes): NNN'.
# macOS BSD time: -l prints '  NNN  maximum resident set size' in BYTES.
# Both write to stderr. We capture stderr into a file and parse after the run.
have_gnu_time() {
  /usr/bin/time --version >/dev/null 2>&1
}

wrap_time() {
  # Prepend the correct time invocation. When neither is available, skip.
  # $@ = the actual command + args.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    /usr/bin/time -l "$@"
  elif have_gnu_time; then
    /usr/bin/time -v "$@"
  else
    "$@"
  fi
}

parse_rss_from_time_stderr() {
  # $1: path to a stderr log produced by wrap_time.
  # Emits the peak RSS in MiB (or empty when none can be parsed).
  local file="$1"
  local kb bytes
  # GNU time: kbytes
  kb="$(grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' "$file" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)"
  if [[ -n "$kb" ]]; then
    awk -v k="$kb" 'BEGIN { printf "%.1f", k / 1024.0 }'
    return
  fi
  # BSD time: bytes on a line ending 'maximum resident set size'
  bytes="$(grep -oE '[0-9]+  *maximum resident set size' "$file" 2>/dev/null | grep -oE '^[0-9]+' | head -1 || true)"
  if [[ -n "$bytes" ]]; then
    awk -v b="$bytes" 'BEGIN { printf "%.1f", b / (1024.0 * 1024.0) }'
    return
  fi
  echo ""
}

# ---- backend extraction from stderr ---------------------------------------
# Every speech engine logs one of:
#   parakeet: using CUDA backend (RTX 4090)
#   tts-cpp: using Vulkan backend (llvmpipe)
#   audiogen: using Metal backend (Apple M3 Max)
# Whisper's own bench prints 'whisper_backend_init_gpu: using <NAME> backend'
# instead. The regex captures the token between 'using' and 'backend'.
parse_backend_from_stderr() {
  local file="$1"
  local hit
  hit="$(grep -oE 'using [A-Za-z0-9]+ backend' "$file" 2>/dev/null | head -1 | awk '{print $2}' || true)"
  if [[ -n "$hit" ]]; then
    echo "$hit"; return
  fi
  # Whisper's system_info line names the SIMD/vendor but no ggml backend key
  # if the run stayed on CPU — surface CPU explicitly to avoid 'unknown'.
  if grep -q "system_info: n_threads" "$file" 2>/dev/null && \
     ! grep -q "GPU\|CUDA\|Metal\|Vulkan" "$file" 2>/dev/null; then
    echo "CPU"; return
  fi
  echo ""
}

# ---- native bench: one invocation, parse the emitted JSON ------------------
run_native() {
  local native_json="$1"; shift    # path we ask the bench to write
  local stderr_log="$1"; shift
  local rss_log="$1"; shift

  # Combine bench stderr and time stderr into one file; time's output starts
  # after the child exits so it doesn't interleave with bench log lines.
  if wrap_time "$BINARY" $EXPANDED_ARGS --runs "$RUNS" --warmup "$WARMUP" --json-out "$native_json" \
      > "$stderr_log.stdout" 2> "$rss_log"; then
    cat "$rss_log" >> "$stderr_log"          # for backend parsing
  else
    cat "$rss_log" >> "$stderr_log"
    return 1
  fi

  if ! [[ -s "$native_json" ]]; then
    echo "native bench did not emit --json-out file" >&2
    return 1
  fi

  # Native bench JSON keys we care about: top-level `rtf.median`, and the
  # 'total' / 'e2e' / 'tot' stage median_ms. Try the common names in order.
  local median wmin wmax rtf
  median="$(jq -r '(.stages.tot.median_ms // .stages.total.median_ms // .stages.e2e.median_ms // empty)' "$native_json" 2>/dev/null || true)"
  wmin="$(jq   -r '(.stages.tot.min_ms    // .stages.total.min_ms    // .stages.e2e.min_ms    // empty)' "$native_json" 2>/dev/null || true)"
  wmax="$(jq   -r '(.stages.tot.max_ms    // .stages.total.max_ms    // .stages.e2e.max_ms    // empty)' "$native_json" 2>/dev/null || true)"
  rtf="$(jq    -r '(.rtf.median // empty)'                                                                  "$native_json" 2>/dev/null || true)"

  # Echo back "median|min|max|rtf" (empty fields ok) for the caller to consume.
  echo "${median}|${wmin}|${wmax}|${rtf}"
}

# ---- time-wrapped bench: N invocations, we take the median ------------------
run_one_time_wrapped() {
  local iter="$1" stderr_log="$2" rss_log="$3"
  local start_ns end_ns
  start_ns="$(date +%s%N)"
  # shellcheck disable=SC2086
  if ! wrap_time "$BINARY" $EXPANDED_ARGS > "$stderr_log.stdout" 2> "$rss_log"; then
    tail -20 "$rss_log" >&2
    return 1
  fi
  end_ns="$(date +%s%N)"
  cat "$rss_log" >> "$stderr_log"
  awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.1f", (e - s) / 1000000.0 }'
}

# ---- run --------------------------------------------------------------------
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

case "$BENCH_KIND" in
  native)
    native_json="$tmp_dir/native.json"
    stderr_log="$tmp_dir/stderr.log"
    rss_log="$tmp_dir/rss.log"
    parsed=""
    if ! parsed="$(run_native "$native_json" "$stderr_log" "$rss_log")"; then
      BACKEND="$(parse_backend_from_stderr "$stderr_log")"
      BACKEND="${BACKEND:-unknown}"
      emit_json "run-failed" null null null " (native bench invocation failed)"
      exit 0
    fi
    IFS='|' read -r n_med n_min n_max n_rtf <<< "$parsed"
    [[ -n "$n_med" ]] || n_med="null"
    [[ -n "$n_min" ]] || n_min="null"
    [[ -n "$n_max" ]] || n_max="null"
    if [[ -n "$n_rtf" ]]; then RTF_MEDIAN="$n_rtf"; fi

    BACKEND="$(parse_backend_from_stderr "$stderr_log")"
    BACKEND="${BACKEND:-unknown}"
    rss="$(parse_rss_from_time_stderr "$stderr_log")"
    [[ -n "$rss" ]] && PEAK_RSS_MIB="$rss"
    emit_json "ok" "$n_med" "$n_min" "$n_max"
    ;;

  time-wrapped)
    for i in $(seq 1 "$WARMUP"); do
      echo "warmup $i/$WARMUP" >&2
      wu_stderr="$tmp_dir/warmup-$i.err"
      wu_rss="$tmp_dir/warmup-$i.rss"
      if ! run_one_time_wrapped "$i" "$wu_stderr" "$wu_rss" >/dev/null; then
        BACKEND="$(parse_backend_from_stderr "$wu_stderr")"; BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (warmup failed)"; exit 0
      fi
    done

    declare -a wall_ms=()
    max_rss_seen=""
    combined_stderr="$tmp_dir/combined.err"
    : > "$combined_stderr"
    for i in $(seq 1 "$RUNS"); do
      echo "run $i/$RUNS" >&2
      r_stderr="$tmp_dir/run-$i.err"
      r_rss="$tmp_dir/run-$i.rss"
      if ! ms="$(run_one_time_wrapped "$i" "$r_stderr" "$r_rss")"; then
        cat "$r_stderr" >> "$combined_stderr"
        BACKEND="$(parse_backend_from_stderr "$combined_stderr")"; BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (timed run $i failed)"; exit 0
      fi
      wall_ms+=("$ms")
      cat "$r_stderr" >> "$combined_stderr"
      rss_i="$(parse_rss_from_time_stderr "$r_stderr")"
      if [[ -n "$rss_i" ]]; then
        if [[ -z "$max_rss_seen" ]] || awk -v a="$rss_i" -v b="$max_rss_seen" 'BEGIN{exit !(a>b)}'; then
          max_rss_seen="$rss_i"
        fi
      fi
    done

    stats="$(printf '%s\n' "${wall_ms[@]}" | jq -s '{
      median: (sort | if length%2==1 then .[length/2|floor] else (.[length/2-1] + .[length/2]) / 2 end),
      min: min,
      max: max
    }')"
    med="$(echo "$stats" | jq '.median')"
    mn="$(echo "$stats" | jq '.min')"
    mx="$(echo "$stats" | jq '.max')"

    BACKEND="$(parse_backend_from_stderr "$combined_stderr")"
    BACKEND="${BACKEND:-unknown}"
    if [[ -n "$max_rss_seen" ]]; then PEAK_RSS_MIB="$max_rss_seen"; fi

    # Compute RTF only when families.json declared audio_duration_seconds.
    if [[ "$AUDIO_DURATION_S" != "null" && -n "$AUDIO_DURATION_S" ]]; then
      RTF_MEDIAN="$(awk -v m="$med" -v s="$AUDIO_DURATION_S" 'BEGIN { printf "%.3f", m / (s * 1000.0) }')"
    fi

    emit_json "ok" "$med" "$mn" "$mx"
    ;;

  *)
    emit_json "run-failed" null null null " (unknown bench_kind '$BENCH_KIND')"
    exit 1
    ;;
esac
