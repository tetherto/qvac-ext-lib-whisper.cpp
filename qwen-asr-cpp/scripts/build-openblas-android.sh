#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENBLAS_VERSION="${OPENBLAS_VERSION:-0.3.33}"
OPENBLAS_URL="https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz"
OPENBLAS_SHA256="${OPENBLAS_SHA256:-6761af1d9f5d353ab4f0b7497be2643313b36c8f31caec0144bfef198e71e6ab}"

VENDOR_ROOT="${ROOT}/vendor/openblas-android"
SRC_DIR="${VENDOR_ROOT}/src/OpenBLAS-${OPENBLAS_VERSION}"
TARBALL="${VENDOR_ROOT}/src/OpenBLAS-${OPENBLAS_VERSION}.tar.gz"
INSTALL_ROOT="${VENDOR_ROOT}/install"

ANDROID_API="${ANDROID_API:-24}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"

: "${ANDROID_NDK_HOME:=}"
ndk_is_valid() { [[ -n "$1" && -f "$1/build/cmake/android.toolchain.cmake" ]]; }
resolve_ndk() {
    if ndk_is_valid "${ANDROID_NDK_HOME}"; then return; fi
    if ndk_is_valid "${ANDROID_NDK_ROOT:-}"; then
        ANDROID_NDK_HOME="${ANDROID_NDK_ROOT}"; return
    fi
    local candidate
    for base in "${HOME}/Library/Android/sdk/ndk" "${HOME}/Android/Sdk/ndk" "/opt/android-sdk/ndk" "/usr/local/lib/android/sdk/ndk"; do
        if [[ -d "${base}" ]]; then
            candidate="$(ls -d "${base}"/* 2>/dev/null | sort -V | tail -1)"
            if ndk_is_valid "${candidate}"; then
                ANDROID_NDK_HOME="${candidate}"; return
            fi
        fi
    done
    ANDROID_NDK_HOME=""
}
resolve_ndk

log() { printf '\033[1;34m[build-openblas-android]\033[0m %s\n' "$*"; }

require_ndk() {
    if [[ -z "${ANDROID_NDK_HOME}" ]]; then
        echo "[build-openblas-android] ANDROID_NDK_HOME not resolved." >&2
        exit 1
    fi
    log "NDK=${ANDROID_NDK_HOME}"
}

host_tag() {
    case "$(uname -s)" in
        Darwin) echo "darwin-x86_64" ;;
        Linux)  echo "linux-x86_64" ;;
        *) echo "[build-openblas-android] unsupported host: $(uname -s)" >&2; exit 1 ;;
    esac
}

download_openblas() {
    mkdir -p "${VENDOR_ROOT}/src"
    if [[ -f "${TARBALL}" ]]; then
        log "tarball already present: ${TARBALL}"
    else
        log "downloading ${OPENBLAS_URL}"
        curl -fL --retry 3 --retry-delay 2 -o "${TARBALL}.partial" "${OPENBLAS_URL}"
        mv "${TARBALL}.partial" "${TARBALL}"
    fi
    log "verifying sha256"
    local actual
    actual="$(shasum -a 256 "${TARBALL}" | awk '{print $1}')"
    if [[ "${actual}" != "${OPENBLAS_SHA256}" ]]; then
        echo "[build-openblas-android] sha256 mismatch: got ${actual}, expected ${OPENBLAS_SHA256}" >&2
        exit 1
    fi
}

extract_openblas() {
    if [[ -d "${SRC_DIR}" ]]; then
        log "src tree already extracted at ${SRC_DIR}"
    else
        log "extracting tarball"
        tar -xzf "${TARBALL}" -C "${VENDOR_ROOT}/src"
    fi
}

ndk_tool() {
    local kind="$1"
    local triple="$2"
    local api="$3"
    local bin_dir="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/$(host_tag)/bin"
    case "${kind}" in
        cc)     echo "${bin_dir}/${triple}${api}-clang" ;;
        ar)     echo "${bin_dir}/llvm-ar" ;;
        ranlib) echo "${bin_dir}/llvm-ranlib" ;;
        strip)  echo "${bin_dir}/llvm-strip" ;;
        *) echo "[build-openblas-android] unknown ndk_tool kind: ${kind}" >&2; exit 1 ;;
    esac
}

build_one_abi() {
    local abi="$1"
    local target
    local triple
    local extra_cflags=""
    case "${abi}" in
        arm64-v8a)
            target="ARMV8"
            triple="aarch64-linux-android"
            ;;
        x86_64)
            target="HASWELL"
            triple="x86_64-linux-android"
            extra_cflags="-msse4.2"
            ;;
        *)
            echo "[build-openblas-android] unsupported abi: ${abi}" >&2
            exit 1
            ;;
    esac

    local install_dir="${INSTALL_ROOT}/${abi}"
    if [[ -f "${install_dir}/lib/libopenblas.a" ]]; then
        log "skip ${abi}: already built at ${install_dir}/lib/libopenblas.a"
        return
    fi

    log "building OpenBLAS ${OPENBLAS_VERSION} for android-${abi} (TARGET=${target})"
    local cc
    cc="$(ndk_tool cc "${triple}" "${ANDROID_API}")"
    local ar
    ar="$(ndk_tool ar "${triple}" "${ANDROID_API}")"
    local ranlib
    ranlib="$(ndk_tool ranlib "${triple}" "${ANDROID_API}")"

    local build_dir="${VENDOR_ROOT}/build/${abi}"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cp -r "${SRC_DIR}"/. "${build_dir}/"

    pushd "${build_dir}" >/dev/null
    make clean >/dev/null 2>&1 || true
    make -j "${PARALLEL_JOBS}" \
        TARGET="${target}" \
        BINARY=64 \
        HOSTCC=clang \
        CC="${cc}" \
        AR="${ar}" \
        RANLIB="${ranlib}" \
        CFLAGS="-O3 -fPIC ${extra_cflags}" \
        NOFORTRAN=1 \
        NO_SHARED=1 \
        ONLY_CBLAS=1 \
        USE_OPENMP=0 \
        USE_THREAD=1 \
        NUM_THREADS=16
    make install \
        PREFIX="${install_dir}" \
        NO_SHARED=1
    popd >/dev/null

    log "installed ${abi} -> ${install_dir}"
}

main() {
    local abis=("$@")
    if [[ ${#abis[@]} -eq 0 ]]; then
        abis=("arm64-v8a" "x86_64")
    fi
    require_ndk
    download_openblas
    extract_openblas
    for abi in "${abis[@]}"; do
        build_one_abi "${abi}"
    done
    log "done"
}

main "$@"
