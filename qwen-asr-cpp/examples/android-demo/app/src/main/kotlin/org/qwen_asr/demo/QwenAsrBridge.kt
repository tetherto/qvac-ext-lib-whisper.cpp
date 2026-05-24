package org.qwen_asr.demo

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

class QwenAsrEngine(modelDir: String, nThreads: Int = 0) : AutoCloseable {
    private var handle: Long = QwenAsrBridge.nativeCreate(modelDir, nThreads)

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

    @JvmStatic external fun nativeCreate(modelDir: String, nThreads: Int): Long
    @JvmStatic external fun nativeDestroy(handle: Long)
    @JvmStatic external fun nativeTranscribe(handle: Long, wavPath: String): QwenAsrResult
}
