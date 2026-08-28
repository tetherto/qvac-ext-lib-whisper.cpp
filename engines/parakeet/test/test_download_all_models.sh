#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SCRIPT="$SCRIPT_DIR/../scripts/download-all-models.sh"
WORK_DIR="$(mktemp -d)"
STUB_DIR="$WORK_DIR/bin"
DOWNLOAD_LOG="$WORK_DIR/downloads.log"
EXPECTED_URL="https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
EXPECTED_MODEL="$WORK_DIR/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"

cleanup() {
  rm -rf "$WORK_DIR"
}

create_curl_stub() {
  mkdir -p "$STUB_DIR"
  cat > "$STUB_DIR/curl" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

parse_arguments() {
  while (( $# > 0 )); do
    case "$1" in
      -o)
        destination="$2"
        shift 2
        ;;
      http*)
        url="$1"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
}

destination=
url=
parse_arguments "$@"
printf '%s\n' "$url" >> "$DOWNLOAD_LOG"
: > "$destination"
EOF
  chmod +x "$STUB_DIR/curl"
}

prepare_script() {
  mkdir -p "$WORK_DIR/scripts"
  cp "$SOURCE_SCRIPT" "$WORK_DIR/scripts/download-all-models.sh"
}

assert_download() {
  grep -Fxq "$EXPECTED_URL" "$DOWNLOAD_LOG"
  test -f "$EXPECTED_MODEL"
}

trap cleanup EXIT
create_curl_stub
prepare_script
PATH="$STUB_DIR:$PATH" DOWNLOAD_LOG="$DOWNLOAD_LOG" \
  bash "$WORK_DIR/scripts/download-all-models.sh" > /dev/null
assert_download
