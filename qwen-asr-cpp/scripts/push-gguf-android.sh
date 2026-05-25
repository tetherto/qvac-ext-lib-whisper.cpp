#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PKG="${PKG:-org.qwen_asr.demo}"
GGUF_FILE="${GGUF_FILE:-qwen3-asr-0.6b.gguf}"
SRC_PATH="${SRC_PATH:-${ROOT}/models/gguf/${GGUF_FILE}}"
TMP_DIR="${TMP_DIR:-/data/local/tmp/qwen-asr-push}"

log() { printf '\033[1;34m[push-gguf-android]\033[0m %s\n' "$*"; }

ensure_adb_device() {
    local count
    count="$(adb devices | awk 'NR>1 && $2=="device" {n++} END {print n+0}')"
    if [[ "${count}" -eq 0 ]]; then
        echo "[push-gguf-android] no adb device. Boot the emulator or plug in a phone." >&2
        exit 1
    fi
    if [[ -n "${ANDROID_SERIAL:-}" ]]; then
        log "using ANDROID_SERIAL=${ANDROID_SERIAL}"
    fi
}

ensure_app_installed() {
    if ! adb shell pm path "${PKG}" >/dev/null 2>&1; then
        echo "[push-gguf-android] app '${PKG}' is not installed. Build & install the apk first:" >&2
        echo "    cd native-app/android && gradle :app:assembleDebug" >&2
        echo "    adb install -r app/build/outputs/apk/debug/app-debug.apk" >&2
        exit 1
    fi
}

ensure_src_exists() {
    if [[ ! -f "${SRC_PATH}" ]]; then
        echo "[push-gguf-android] missing source: ${SRC_PATH}" >&2
        echo "Generate it first:" >&2
        echo "    .venv-convert/bin/python scripts/convert-hf-to-gguf.py models/hf/0.6b ${SRC_PATH}" >&2
        exit 1
    fi
}

reset_tmp_dir() {
    adb shell rm -rf "${TMP_DIR}"
    adb shell mkdir -p "${TMP_DIR}"
}

push_to_tmp() {
    log "push -> tmp: ${GGUF_FILE} ($(stat -f %z "${SRC_PATH}" 2>/dev/null || stat -c %s "${SRC_PATH}") bytes)"
    adb push "${SRC_PATH}" "${TMP_DIR}/${GGUF_FILE}" </dev/null
    adb shell chmod 644 "${TMP_DIR}/${GGUF_FILE}" </dev/null
}

move_into_app() {
    log "run-as cp: ${GGUF_FILE}"
    adb shell run-as "${PKG}" cp "${TMP_DIR}/${GGUF_FILE}" "files/${GGUF_FILE}" </dev/null
    adb shell rm -f "${TMP_DIR}/${GGUF_FILE}" </dev/null
}

verify_destination() {
    log "destination listing:"
    adb shell run-as "${PKG}" ls -la "files/${GGUF_FILE}"
}

main() {
    ensure_adb_device
    ensure_app_installed
    ensure_src_exists
    reset_tmp_dir
    push_to_tmp
    move_into_app
    verify_destination
    log "done"
}

main "$@"
