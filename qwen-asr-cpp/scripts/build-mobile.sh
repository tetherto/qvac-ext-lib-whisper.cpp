#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_ROOT="${ROOT}/build-mobile"

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

: "${ANDROID_PLATFORM:=android-24}"

: "${IOS_DEPLOYMENT_TARGET:=16.4}"

: "${CMAKE_BUILD_TYPE:=Release}"
: "${PARALLEL_JOBS:=8}"

log() { printf '\033[1;34m[build-mobile]\033[0m %s\n' "$*"; }

require_tool() {
    local tool="$1"
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[build-mobile] missing required tool: ${tool}" >&2
        exit 1
    fi
}

build_android() {
    local abi="$1"
    local build_dir="${BUILD_ROOT}/android-${abi}"
    local page_flags="-Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384"
    local openblas_lib="${ROOT}/vendor/openblas-android/install/${abi}/lib/libopenblas.a"
    if [[ ! -f "${openblas_lib}" ]]; then
        log "OpenBLAS prebuilt missing for ${abi}; building it now"
        bash "${SCRIPT_DIR}/build-openblas-android.sh" "${abi}"
    fi
    log "Android ${abi} (NDK=${ANDROID_NDK_HOME})"
    rm -rf "${build_dir}"
    cmake -S "${ROOT}" -B "${build_dir}" \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="${abi}" \
        -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
        -DCMAKE_SHARED_LINKER_FLAGS="${page_flags}" \
        -DCMAKE_EXE_LINKER_FLAGS="${page_flags}" \
        -DQWEN_USE_BLAS=ON \
        -DQWEN_BUILD_EXECUTABLES=OFF \
        -DQWEN_BUILD_TESTS=OFF
    cmake --build "${build_dir}" -j "${PARALLEL_JOBS}"
}

build_ios() {
    local sysroot="$1"
    local subdir="$2"
    local build_dir="${BUILD_ROOT}/ios-${subdir}"
    log "iOS ${sysroot} arm64 (deployment ${IOS_DEPLOYMENT_TARGET})"
    rm -rf "${build_dir}"
    cmake -S "${ROOT}" -B "${build_dir}" \
        -G Xcode \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_SYSROOT="${sysroot}" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="${IOS_DEPLOYMENT_TARGET}" \
        -DQWEN_USE_BLAS=ON \
        -DQWEN_BUILD_EXECUTABLES=OFF \
        -DQWEN_BUILD_TESTS=OFF
    cmake --build "${build_dir}" --config "${CMAKE_BUILD_TYPE}" -- -quiet
}

build_xcframework() {
    local out="${BUILD_ROOT}/QwenAsr.xcframework"
    local device_lib="${BUILD_ROOT}/ios-device/${CMAKE_BUILD_TYPE}-iphoneos/libqwen-asr.a"
    local sim_lib="${BUILD_ROOT}/ios-simulator/${CMAKE_BUILD_TYPE}-iphonesimulator/libqwen-asr.a"
    local device_vendor="${BUILD_ROOT}/ios-device/${CMAKE_BUILD_TYPE}-iphoneos/libqwen-asr-vendor.a"
    local sim_vendor="${BUILD_ROOT}/ios-simulator/${CMAKE_BUILD_TYPE}-iphonesimulator/libqwen-asr-vendor.a"

    if [[ ! -f "${device_lib}" || ! -f "${sim_lib}" ]]; then
        echo "[build-mobile] iOS .a libraries not found; run iOS builds first" >&2
        exit 1
    fi

    local merge_dir="${BUILD_ROOT}/ios-merged"
    rm -rf "${merge_dir}"
    mkdir -p "${merge_dir}/iphoneos" "${merge_dir}/iphonesimulator"
    libtool -static -o "${merge_dir}/iphoneos/libqwen-asr-combined.a"        "${device_lib}" "${device_vendor}"
    libtool -static -o "${merge_dir}/iphonesimulator/libqwen-asr-combined.a" "${sim_lib}"    "${sim_vendor}"

    rm -rf "${out}"
    xcodebuild -create-xcframework \
        -library "${merge_dir}/iphoneos/libqwen-asr-combined.a"        -headers "${ROOT}/include" \
        -library "${merge_dir}/iphonesimulator/libqwen-asr-combined.a" -headers "${ROOT}/include" \
        -output "${out}"
    log "xcframework written: ${out}"
}

main() {
    local targets=("$@")
    if [[ ${#targets[@]} -eq 0 ]]; then
        targets=("android-arm64" "android-x86_64" "ios-device" "ios-simulator" "xcframework")
    fi

    require_tool cmake
    mkdir -p "${BUILD_ROOT}"

    for t in "${targets[@]}"; do
        case "${t}" in
            android-arm64)
                if [[ -z "${ANDROID_NDK_HOME}" ]]; then
                    echo "[build-mobile] ANDROID_NDK_HOME not set; skipping ${t}" >&2
                    continue
                fi
                build_android arm64-v8a
                ;;
            android-x86_64)
                if [[ -z "${ANDROID_NDK_HOME}" ]]; then
                    echo "[build-mobile] ANDROID_NDK_HOME not set; skipping ${t}" >&2
                    continue
                fi
                build_android x86_64
                ;;
            ios-device)
                require_tool xcodebuild
                build_ios iphoneos device
                ;;
            ios-simulator)
                require_tool xcodebuild
                build_ios iphonesimulator simulator
                ;;
            xcframework)
                require_tool xcodebuild
                build_xcframework
                ;;
            *)
                echo "[build-mobile] unknown target: ${t}" >&2
                exit 1
                ;;
        esac
    done

    log "done"
}

main "$@"
