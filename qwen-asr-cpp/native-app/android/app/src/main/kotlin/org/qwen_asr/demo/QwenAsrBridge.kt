package org.qwen_asr.demo

enum class QwenAsrBackend(val ordinalValue: Int, val displayName: String) {
    SAFETENSORS(0, "Safetensors (v0.1)"),
    GGUF(1, "GGUF (v0.2)");

    val isAvailable: Boolean
        get() = QwenAsrBridge.nativeBackendAvailable(ordinalValue) != 0
}

data class QwenAsrResult(
    val text: String,
    val encodeMs: Double,
    val decodeMs: Double,
    val totalMs: Double,
    val audioMs: Double,
    val tokens: Int,
) {
    val realTimeFactor: Double get() = if (totalMs > 0.0) audioMs / totalMs else 0.0
}

class QwenAsrEngine(
    modelPath: String,
    val backend: QwenAsrBackend = QwenAsrBackend.SAFETENSORS,
    nThreads: Int = 0,
) : AutoCloseable {
    private var handle: Long =
        QwenAsrBridge.nativeCreate(modelPath, nThreads, backend.ordinalValue)

    fun transcribe(wavPath: String): QwenAsrResult =
        QwenAsrBridge.nativeTranscribe(handle, wavPath)

    override fun close() {
        if (handle != 0L) {
            QwenAsrBridge.nativeDestroy(handle)
            handle = 0L
        }
    }
}

object QwenAsrBridge {
    init {
        System.loadLibrary("qwen_jni")
    }

    @JvmStatic external fun nativeCreate(modelPath: String, nThreads: Int, backendOrdinal: Int): Long
    @JvmStatic external fun nativeDestroy(handle: Long)
    @JvmStatic external fun nativeTranscribe(handle: Long, wavPath: String): QwenAsrResult
    @JvmStatic external fun nativeBackendAvailable(backendOrdinal: Int): Int
}
