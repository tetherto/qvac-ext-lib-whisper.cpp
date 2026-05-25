cmake_minimum_required(VERSION 3.20)
project(qwen-asr-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(QWEN_STANDALONE_DEFAULT ON)
else()
    set(QWEN_STANDALONE_DEFAULT OFF)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(QWEN_MOBILE_DEFAULT OFF)
else()
    set(QWEN_MOBILE_DEFAULT ${QWEN_STANDALONE_DEFAULT})
endif()

option(QWEN_BUILD_LIBRARY        "qwen-asr: build the qwen-asr library"                                ON)
option(QWEN_BUILD_EXECUTABLES    "qwen-asr: build the qwen-asr CLI"                                    ${QWEN_MOBILE_DEFAULT})
option(QWEN_BUILD_TESTS          "qwen-asr: build the integration tests"                               ${QWEN_MOBILE_DEFAULT})
option(QWEN_INSTALL              "qwen-asr: generate install rules"                                    ON)
option(QWEN_USE_BLAS             "qwen-asr: enable BLAS-accelerated matmul (Accelerate / OpenBLAS)"    ON)
option(QWEN_CCACHE               "qwen-asr: use ccache as compiler launcher if available"              ON)
option(QWEN_BACKEND_SAFETENSORS  "qwen-asr: build the safetensors backend (antirez/qwen-asr, v0.1)"    ON)
option(QWEN_BACKEND_GGUF         "qwen-asr: build the GGUF/GGML backend (v0.2 -- in progress)"         OFF)

if (NOT QWEN_BACKEND_SAFETENSORS AND NOT QWEN_BACKEND_GGUF)
    message(FATAL_ERROR
        "qwen-asr: at least one backend must be enabled "
        "(set -DQWEN_BACKEND_SAFETENSORS=ON or -DQWEN_BACKEND_GGUF=ON).")
endif()

if (QWEN_CCACHE)
    find_program(QWEN_CCACHE_PROGRAM ccache)
    if (QWEN_CCACHE_PROGRAM)
        message(STATUS "qwen-asr: using ccache at ${QWEN_CCACHE_PROGRAM}")
    endif()
endif()

function(qwen_apply_ccache target)
    if (QWEN_CCACHE_PROGRAM)
        set_target_properties(${target} PROPERTIES
            C_COMPILER_LAUNCHER   "${QWEN_CCACHE_PROGRAM}"
            CXX_COMPILER_LAUNCHER "${QWEN_CCACHE_PROGRAM}"
        )
    endif()
endfunction()

if (QWEN_BACKEND_SAFETENSORS)
set(QWEN_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/qwen-asr-c)

set(QWEN_VENDOR_SOURCES
    ${QWEN_VENDOR_DIR}/qwen_asr.c
    ${QWEN_VENDOR_DIR}/qwen_asr_audio.c
    ${QWEN_VENDOR_DIR}/qwen_asr_decoder.c
    ${QWEN_VENDOR_DIR}/qwen_asr_encoder.c
    ${QWEN_VENDOR_DIR}/qwen_asr_kernels.c
    ${QWEN_VENDOR_DIR}/qwen_asr_kernels_generic.c
    ${QWEN_VENDOR_DIR}/qwen_asr_safetensors.c
    ${QWEN_VENDOR_DIR}/qwen_asr_tokenizer.c
)

set(QWEN_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
if (DEFINED ANDROID_ABI AND NOT ANDROID_ABI STREQUAL "")
    set(QWEN_TARGET_ARCH "${ANDROID_ABI}")
elseif (DEFINED CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
    set(QWEN_TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
endif()

if (QWEN_TARGET_ARCH MATCHES "(x86_64|AMD64|x64)")
    list(APPEND QWEN_VENDOR_SOURCES ${QWEN_VENDOR_DIR}/qwen_asr_kernels_avx.c)
elseif (QWEN_TARGET_ARCH MATCHES "(aarch64|arm64|arm64-v8a)")
    list(APPEND QWEN_VENDOR_SOURCES ${QWEN_VENDOR_DIR}/qwen_asr_kernels_neon.c)
endif()
message(STATUS "qwen-asr: target arch resolved to '${QWEN_TARGET_ARCH}'")

add_library(qwen-asr-vendor STATIC ${QWEN_VENDOR_SOURCES})
target_include_directories(qwen-asr-vendor PUBLIC
    $<BUILD_INTERFACE:${QWEN_VENDOR_DIR}>
)
set_target_properties(qwen-asr-vendor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
)
qwen_apply_ccache(qwen-asr-vendor)

if (MSVC)
    target_compile_options(qwen-asr-vendor PRIVATE /O2 /fp:fast /W3)
    target_compile_definitions(qwen-asr-vendor PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
    )
else()
    target_compile_options(qwen-asr-vendor PRIVATE
        -O3
        -ffast-math
        -funroll-loops
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
        -Wno-unused-but-set-variable
        -Wno-unused-variable
    )
    if (NOT CMAKE_CROSSCOMPILING AND (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86|arm|aarch64|arm64)"))
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-march=native" QWEN_HAS_MARCH_NATIVE)
        if (QWEN_HAS_MARCH_NATIVE)
            target_compile_options(qwen-asr-vendor PRIVATE -march=native)
        endif()
    endif()
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (ANDROID_ABI STREQUAL "armeabi-v7a" OR ANDROID_ABI STREQUAL "x86")
        message(FATAL_ERROR
            "qwen-asr: 32-bit Android ABI '${ANDROID_ABI}' is not supported. "
            "The vendored AArch64 NEON kernels and the working-set size of Qwen3-ASR "
            "(>=1 GiB at q4_0) require a 64-bit ABI. "
            "Build for arm64-v8a (device) or x86_64 (emulator).")
    endif()
    find_package(Threads REQUIRED)
    target_link_libraries(qwen-asr-vendor PUBLIC Threads::Threads m log)
    if (QWEN_USE_BLAS)
        set(QWEN_OPENBLAS_ANDROID_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/openblas-android/install/${ANDROID_ABI}" CACHE PATH
            "qwen-asr: OpenBLAS prebuilt root for Android ABI (see scripts/build-openblas-android.sh)")
        set(QWEN_OPENBLAS_ANDROID_LIB "${QWEN_OPENBLAS_ANDROID_ROOT}/lib/libopenblas.a")
        set(QWEN_OPENBLAS_ANDROID_INC "${QWEN_OPENBLAS_ANDROID_ROOT}/include")
        if (EXISTS "${QWEN_OPENBLAS_ANDROID_LIB}" AND EXISTS "${QWEN_OPENBLAS_ANDROID_INC}/cblas.h")
            target_compile_definitions(qwen-asr-vendor PRIVATE USE_BLAS USE_OPENBLAS)
            target_include_directories(qwen-asr-vendor PRIVATE "${QWEN_OPENBLAS_ANDROID_INC}")
            target_link_libraries(qwen-asr-vendor PUBLIC "${QWEN_OPENBLAS_ANDROID_LIB}")
            message(STATUS "qwen-asr: Android target -- linking OpenBLAS from ${QWEN_OPENBLAS_ANDROID_LIB}")
        else()
            message(STATUS "qwen-asr: Android target -- OpenBLAS prebuilt missing at ${QWEN_OPENBLAS_ANDROID_ROOT}; falling back to SIMD-only kernels")
            message(STATUS "qwen-asr: build it with 'bash scripts/build-openblas-android.sh ${ANDROID_ABI}'")
        endif()
    endif()
elseif (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    find_package(Threads REQUIRED)
    target_link_libraries(qwen-asr-vendor PUBLIC Threads::Threads)
    if (QWEN_USE_BLAS)
        find_library(QWEN_ACCELERATE_FRAMEWORK Accelerate)
        if (QWEN_ACCELERATE_FRAMEWORK)
            target_compile_definitions(qwen-asr-vendor PRIVATE
                USE_BLAS
                ACCELERATE_NEW_LAPACK
            )
            target_link_libraries(qwen-asr-vendor PUBLIC ${QWEN_ACCELERATE_FRAMEWORK})
            message(STATUS "qwen-asr: iOS target -- linking Apple Accelerate framework for BLAS")
        endif()
    endif()
elseif (APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(qwen-asr-vendor PUBLIC Threads::Threads)
    if (QWEN_USE_BLAS)
        find_library(QWEN_ACCELERATE_FRAMEWORK Accelerate)
        if (QWEN_ACCELERATE_FRAMEWORK)
            target_compile_definitions(qwen-asr-vendor PRIVATE
                USE_BLAS
                ACCELERATE_NEW_LAPACK
            )
            target_link_libraries(qwen-asr-vendor PUBLIC ${QWEN_ACCELERATE_FRAMEWORK})
            message(STATUS "qwen-asr: macOS target -- linking Apple Accelerate framework for BLAS")
        endif()
    endif()
elseif (WIN32)
    target_compile_definitions(qwen-asr-vendor PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
elseif (UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(qwen-asr-vendor PUBLIC Threads::Threads m)
    if (QWEN_USE_BLAS)
        find_package(BLAS)
        if (BLAS_FOUND)
            target_compile_definitions(qwen-asr-vendor PRIVATE USE_BLAS USE_OPENBLAS)
            target_link_libraries(qwen-asr-vendor PUBLIC ${BLAS_LIBRARIES})
            find_path(QWEN_OPENBLAS_INCLUDE openblas_config.h
                PATHS /usr/include /usr/include/openblas /usr/local/include /usr/local/include/openblas)
            if (QWEN_OPENBLAS_INCLUDE)
                target_include_directories(qwen-asr-vendor PRIVATE ${QWEN_OPENBLAS_INCLUDE})
            endif()
            message(STATUS "qwen-asr: Linux target -- linking ${BLAS_LIBRARIES} for BLAS")
        else()
            message(STATUS "qwen-asr: BLAS not found; falling back to SIMD-only kernels")
        endif()
    endif()
endif()
endif()

set(QWEN_LIB_SOURCES
    src/backend.cpp
    src/engine_facade.cpp
    src/engine_factory.cpp
    src/cli_dispatch.cpp
    src/c_api.cpp
)

if (QWEN_BACKEND_SAFETENSORS)
    list(APPEND QWEN_LIB_SOURCES src/safetensors/engine_safetensors.cpp)
endif()

if (QWEN_BACKEND_GGUF)
    list(APPEND QWEN_LIB_SOURCES
        src/gguf/engine_gguf.cpp
        src/gguf/model_loader.cpp
        src/gguf/tokenizer.cpp
        src/gguf/audio.cpp
        src/gguf/model.cpp
    )
endif()

if (QWEN_BUILD_LIBRARY)
    if (DEFINED BUILD_SHARED_LIBS)
        add_library(qwen-asr ${QWEN_LIB_SOURCES})
    else()
        add_library(qwen-asr STATIC ${QWEN_LIB_SOURCES})
    endif()
    add_library(qwen-asr::qwen-asr ALIAS qwen-asr)
    qwen_apply_ccache(qwen-asr)

    target_include_directories(qwen-asr
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_compile_definitions(qwen-asr PRIVATE
        QWEN_VERSION="${PROJECT_VERSION}"
    )

    if (QWEN_BACKEND_SAFETENSORS)
        target_compile_definitions(qwen-asr PUBLIC QWEN_HAVE_BACKEND_SAFETENSORS)
        target_link_libraries(qwen-asr PRIVATE qwen-asr-vendor)
        target_include_directories(qwen-asr PRIVATE ${QWEN_VENDOR_DIR})
    endif()

    if (QWEN_BACKEND_GGUF)
        target_compile_definitions(qwen-asr PUBLIC QWEN_HAVE_BACKEND_GGUF)
        if (NOT TARGET ggml)
            set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
            set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml ${CMAKE_CURRENT_BINARY_DIR}/_ggml EXCLUDE_FROM_ALL)
        endif()
        target_link_libraries(qwen-asr PRIVATE $<BUILD_INTERFACE:ggml>)
        target_include_directories(qwen-asr PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/include
        )
    endif()

    set_target_properties(qwen-asr PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME qwen-asr
        CXX_VISIBILITY_PRESET    hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION   ${PROJECT_VERSION}
        SOVERSION 0
    )

    get_target_property(_qwen_type qwen-asr TYPE)
    if (_qwen_type STREQUAL "SHARED_LIBRARY")
        target_compile_definitions(qwen-asr PUBLIC  QWEN_SHARED)
        target_compile_definitions(qwen-asr PRIVATE QWEN_BUILD)
    endif()
endif()

if (QWEN_BUILD_EXECUTABLES)
    if (NOT QWEN_BUILD_LIBRARY)
        message(FATAL_ERROR "QWEN_BUILD_EXECUTABLES requires QWEN_BUILD_LIBRARY=ON")
    endif()

    add_executable(qwen-asr-cli src/cli_main.cpp)
    target_link_libraries(qwen-asr-cli PRIVATE qwen-asr)
    set_target_properties(qwen-asr-cli PROPERTIES OUTPUT_NAME qwen-asr)
    qwen_apply_ccache(qwen-asr-cli)
endif()

if (QWEN_BUILD_TESTS)
    enable_testing()
    include(CTest)

    add_executable(test-construct test/test_construct.cpp)
    target_link_libraries(test-construct PRIVATE qwen-asr)
    target_include_directories(test-construct PRIVATE include)
    qwen_apply_ccache(test-construct)
    add_test(NAME test-construct COMMAND $<TARGET_FILE:test-construct>)
    set_property(TEST test-construct PROPERTY LABELS "unit")

    set(QWEN_E2E_WAV    "${CMAKE_CURRENT_SOURCE_DIR}/test/samples/jfk.wav" CACHE PATH "qwen-asr: wav fixture for end-to-end test")
    set(QWEN_E2E_GROUND "${CMAKE_CURRENT_SOURCE_DIR}/test/samples/jfk.txt" CACHE PATH "qwen-asr: ground-truth transcript for end-to-end test")

    add_executable(test-transcribe-jfk test/test_transcribe_jfk.cpp)
    target_link_libraries(test-transcribe-jfk PRIVATE qwen-asr)
    target_include_directories(test-transcribe-jfk PRIVATE include)
    qwen_apply_ccache(test-transcribe-jfk)

    if (QWEN_BACKEND_SAFETENSORS)
        set(QWEN_E2E_SAFETENSORS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/hf/0.6b" CACHE PATH
            "qwen-asr: safetensors model dir for end-to-end test")
        add_test(NAME test-transcribe-jfk-safetensors
                 COMMAND $<TARGET_FILE:test-transcribe-jfk>
                         "${QWEN_E2E_SAFETENSORS_DIR}"
                         "${QWEN_E2E_WAV}"
                         "${QWEN_E2E_GROUND}"
                         "safetensors")
        set_property(TEST test-transcribe-jfk-safetensors PROPERTY LABELS "e2e;safetensors")
        if (NOT EXISTS "${QWEN_E2E_SAFETENSORS_DIR}/model.safetensors")
            set_property(TEST test-transcribe-jfk-safetensors PROPERTY DISABLED TRUE)
            message(STATUS "qwen-asr: test-transcribe-jfk-safetensors disabled (missing ${QWEN_E2E_SAFETENSORS_DIR}/model.safetensors)")
        endif()
    endif()

    if (QWEN_BACKEND_GGUF)
        set(QWEN_E2E_GGUF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/qwen3-asr-0.6b.gguf" CACHE PATH
            "qwen-asr: .gguf model file for end-to-end test")
        add_test(NAME test-transcribe-jfk-gguf
                 COMMAND $<TARGET_FILE:test-transcribe-jfk>
                         "${QWEN_E2E_GGUF_FILE}"
                         "${QWEN_E2E_WAV}"
                         "${QWEN_E2E_GROUND}"
                         "gguf")
        set_property(TEST test-transcribe-jfk-gguf PROPERTY LABELS "e2e;gguf")
        if (NOT EXISTS "${QWEN_E2E_GGUF_FILE}")
            set_property(TEST test-transcribe-jfk-gguf PROPERTY DISABLED TRUE)
            message(STATUS "qwen-asr: test-transcribe-jfk-gguf disabled (missing ${QWEN_E2E_GGUF_FILE})")
        endif()
    endif()
endif()

if (QWEN_INSTALL AND QWEN_BUILD_LIBRARY AND QWEN_BACKEND_GGUF)
    message(STATUS "qwen-asr: skipping CMake install/export rules (QWEN_BACKEND_GGUF=ON pulls non-exported ggml).")
    set(QWEN_INSTALL OFF)
endif()

if (QWEN_INSTALL AND QWEN_BUILD_LIBRARY)
    set(QWEN_INSTALL_TARGETS qwen-asr)
    if (QWEN_BACKEND_SAFETENSORS)
        list(APPEND QWEN_INSTALL_TARGETS qwen-asr-vendor)
    endif()
    install(TARGETS ${QWEN_INSTALL_TARGETS}
        EXPORT qwen-asr-cpp-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/qwen
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )

    set(QWEN_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/qwen-asr-cpp)

    install(EXPORT qwen-asr-cpp-targets
        FILE qwen-asr-cpp-targets.cmake
        NAMESPACE qwen-asr::
        DESTINATION ${QWEN_CONFIG_INSTALL_DIR}
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/qwen-asr-cppConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/qwen-asr-cppConfig.cmake
        INSTALL_DESTINATION ${QWEN_CONFIG_INSTALL_DIR}
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/qwen-asr-cppConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/qwen-asr-cppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/qwen-asr-cppConfigVersion.cmake
        DESTINATION ${QWEN_CONFIG_INSTALL_DIR}
    )
endif()
