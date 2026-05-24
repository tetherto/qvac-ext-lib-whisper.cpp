cmake_minimum_required(VERSION 3.22)
project(qwen_asr_jni LANGUAGES C CXX)

set(QWEN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..)
set(QWEN_INCLUDE ${QWEN_ROOT}/include)

if (ANDROID_ABI STREQUAL "arm64-v8a")
    set(QWEN_PREBUILT_DIR ${QWEN_ROOT}/build-mobile/android-arm64-v8a)
elseif (ANDROID_ABI STREQUAL "x86_64")
    set(QWEN_PREBUILT_DIR ${QWEN_ROOT}/build-mobile/android-x86_64)
else()
    message(FATAL_ERROR "qwen-asr-jni: unsupported Android ABI '${ANDROID_ABI}'. Build for arm64-v8a or x86_64.")
endif()

set(QWEN_LIB         ${QWEN_PREBUILT_DIR}/libqwen-asr.a)
set(QWEN_LIB_VENDOR  ${QWEN_PREBUILT_DIR}/libqwen-asr-vendor.a)
set(QWEN_LIB_OPENBLAS ${QWEN_ROOT}/vendor/openblas-android/install/${ANDROID_ABI}/lib/libopenblas.a)

foreach(p IN ITEMS "${QWEN_LIB}" "${QWEN_LIB_VENDOR}" "${QWEN_LIB_OPENBLAS}")
    if (NOT EXISTS "${p}")
        message(FATAL_ERROR
            "qwen-asr-jni: missing prebuilt '${p}'. "
            "Run 'bash scripts/build-mobile.sh android-arm64 android-x86_64' from qwen-asr-cpp/ first.")
    endif()
endforeach()

add_library(qwen-asr-prebuilt STATIC IMPORTED)
set_target_properties(qwen-asr-prebuilt PROPERTIES IMPORTED_LOCATION ${QWEN_LIB})

add_library(qwen-asr-vendor-prebuilt STATIC IMPORTED)
set_target_properties(qwen-asr-vendor-prebuilt PROPERTIES IMPORTED_LOCATION ${QWEN_LIB_VENDOR})

add_library(openblas-prebuilt STATIC IMPORTED)
set_target_properties(openblas-prebuilt PROPERTIES IMPORTED_LOCATION ${QWEN_LIB_OPENBLAS})

add_library(qwen_jni SHARED qwen_jni.cpp)

target_include_directories(qwen_jni PRIVATE ${QWEN_INCLUDE})

target_link_libraries(qwen_jni PRIVATE
    qwen-asr-prebuilt
    qwen-asr-vendor-prebuilt
    openblas-prebuilt
    log
)

target_link_options(qwen_jni PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,-z,common-page-size=16384"
)
