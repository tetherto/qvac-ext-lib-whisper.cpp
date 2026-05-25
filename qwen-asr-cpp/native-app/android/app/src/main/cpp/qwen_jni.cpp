#include <jni.h>
#include <android/log.h>
#include <qwen/c_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr const char * LOG_TAG = "qwen-asr-jni";

void log_info(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    __android_log_vprint(ANDROID_LOG_INFO, LOG_TAG, fmt, args);
    va_end(args);
}

jstring make_jstring(JNIEnv * env, const char * s) {
    return env->NewStringUTF(s != nullptr ? s : "");
}

void throw_runtime(JNIEnv * env, const char * msg) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(cls, msg != nullptr ? msg : "qwen-asr-jni: unknown error");
}

qwen_engine * as_handle(jlong h) { return reinterpret_cast<qwen_engine *>(static_cast<intptr_t>(h)); }

}

extern "C" {

qwen_c_backend backend_from_jint(jint ordinal) {
    switch (ordinal) {
        case 1:  return QWEN_BACKEND_GGUF;
        case 0:
        default: return QWEN_BACKEND_SAFETENSORS;
    }
}

JNIEXPORT jlong JNICALL
Java_org_qwen_1asr_demo_QwenAsrBridge_nativeCreate(JNIEnv * env, jclass,
                                                   jstring jModelPath, jint nThreads, jint backendOrdinal) {
    const char * cPath = env->GetStringUTFChars(jModelPath, nullptr);
    qwen_c_backend backend = backend_from_jint(backendOrdinal);
    log_info("create: model_path=%s threads=%d backend=%s",
             cPath, (int) nThreads, qwen_c_backend_name(backend));

    qwen_c_options opts = qwen_c_options_default();
    opts.backend   = backend;
    opts.n_threads = static_cast<int>(nThreads);
    opts.verbose   = 1;

    char * err  = nullptr;
    qwen_engine * e = qwen_c_engine_create(cPath, opts, &err);
    env->ReleaseStringUTFChars(jModelPath, cPath);

    if (e == nullptr) {
        std::string msg = err != nullptr ? err : "qwen_c_engine_create returned null";
        qwen_c_string_free(err);
        throw_runtime(env, msg.c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(e);
}

JNIEXPORT jint JNICALL
Java_org_qwen_1asr_demo_QwenAsrBridge_nativeBackendAvailable(JNIEnv *, jclass, jint backendOrdinal) {
    return qwen_c_backend_available(backend_from_jint(backendOrdinal));
}

JNIEXPORT void JNICALL
Java_org_qwen_1asr_demo_QwenAsrBridge_nativeDestroy(JNIEnv *, jclass, jlong handle) {
    if (handle != 0) qwen_c_engine_destroy(as_handle(handle));
}

JNIEXPORT jobject JNICALL
Java_org_qwen_1asr_demo_QwenAsrBridge_nativeTranscribe(JNIEnv * env, jclass,
                                                       jlong handle, jstring jWavPath) {
    if (handle == 0) {
        throw_runtime(env, "engine handle is null");
        return nullptr;
    }

    const char * cWav = env->GetStringUTFChars(jWavPath, nullptr);

    qwen_c_result result = {nullptr, 0.0, 0.0, 0.0, 0.0, 0};
    char * err = nullptr;
    int rc = qwen_c_engine_transcribe(as_handle(handle), cWav, &result, &err);

    env->ReleaseStringUTFChars(jWavPath, cWav);

    if (rc != 0) {
        std::string msg = err != nullptr ? err : "qwen_c_engine_transcribe failed";
        qwen_c_string_free(err);
        throw_runtime(env, msg.c_str());
        return nullptr;
    }

    jclass resultCls = env->FindClass("org/qwen_asr/demo/QwenAsrResult");
    if (resultCls == nullptr) {
        qwen_c_result_free(&result);
        throw_runtime(env, "QwenAsrResult class not found");
        return nullptr;
    }
    jmethodID ctor = env->GetMethodID(resultCls, "<init>", "(Ljava/lang/String;DDDDI)V");
    if (ctor == nullptr) {
        qwen_c_result_free(&result);
        throw_runtime(env, "QwenAsrResult constructor not found");
        return nullptr;
    }

    jstring text = make_jstring(env, result.text);
    jobject obj  = env->NewObject(resultCls, ctor,
                                  text,
                                  static_cast<jdouble>(result.encode_ms),
                                  static_cast<jdouble>(result.decode_ms),
                                  static_cast<jdouble>(result.total_ms),
                                  static_cast<jdouble>(result.audio_ms),
                                  static_cast<jint>(result.text_tokens));
    qwen_c_result_free(&result);
    return obj;
}

}
