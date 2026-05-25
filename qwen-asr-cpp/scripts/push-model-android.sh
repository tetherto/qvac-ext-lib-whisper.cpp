#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PKG="${PKG:-org.qwen_asr.demo}"
SUBDIR="${SUBDIR:-qwen-0.6b}"
SRC_DIR="${SRC_DIR:-${ROOT}/models/hf/0.6b}"
TMP_DIR="${TMP_DIR:-/data/local/tmp/qwen-asr-push}"

log() { printf '\033[1;34m[push-model-android]\033[0m %s\n' "$*"; }

ensure_adb_device() {
    local count
    count="$(adb devices | awk 'NR>1 && $2=="device" {n++} END {print n+0}')"
    if [[ "${count}" -eq 0 ]]; then
        echo "[push-model-android] no adb device. Boot the emulator or plug in a phone." >&2
        exit 1
    fi
    if [[ -n "${ANDROID_SERIAL:-}" ]]; then
        log "using ANDROID_SERIAL=${ANDROID_SERIAL}"
    fi
}

ensure_app_installed() {
    if ! adb shell pm path "${PKG}" >/dev/null 2>&1; then
        echo "[push-model-android] app '${PKG}' is not installed. Build & install the apk first:" >&2
        echo "    cd native-app/android && gradle :app:assembleDebug" >&2
        echo "    adb install -r app/build/outputs/apk/debug/app-debug.apk" >&2
        exit 1
    fi
}

reset_tmp_dir() {
    adb shell rm -rf "${TMP_DIR}"
    adb shell mkdir -p "${TMP_DIR}"
}

ensure_dest_dir() {
    adb shell run-as "${PKG}" mkdir -p "files/${SUBDIR}"
}

push_file_to_tmp() {
    local f="$1"
    log "push -> tmp: ${f}"
    adb push "${SRC_DIR}/${f}" "${TMP_DIR}/${f}" >/dev/null </dev/null
    adb shell chmod 644 "${TMP_DIR}/${f}" </dev/null
}

move_file_to_app() {
    local f="$1"
    log "run-as cp: ${f}"
    adb shell run-as "${PKG}" cp "${TMP_DIR}/${f}" "files/${SUBDIR}/${f}" </dev/null
    adb shell rm -f "${TMP_DIR}/${f}" </dev/null
}

provision_file() {
    local f="$1"
    push_file_to_tmp "${f}"
    move_file_to_app "${f}"
}

list_required_files() {
    cat <<'EOF'
model.safetensors
vocab.json
merges.txt
config.json
generation_config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
EOF
}

provision_all_files() {
    while IFS= read -r f; do
        if [[ -f "${SRC_DIR}/${f}" ]]; then
            provision_file "${f}"
        else
            log "skip missing source: ${f}"
        fi
    done < <(list_required_files)
}

cleanup_tmp_dir() {
    adb shell rm -rf "${TMP_DIR}"
}

verify_destination() {
    log "destination listing:"
    adb shell run-as "${PKG}" ls -la "files/${SUBDIR}"
}

main() {
    ensure_adb_device
    ensure_app_installed
    reset_tmp_dir
    ensure_dest_dir
    provision_all_files
    cleanup_tmp_dir
    verify_destination
    log "done"
}

main "$@"
