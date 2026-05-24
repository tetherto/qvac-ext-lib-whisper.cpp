package org.qwen_asr.demo

import android.content.Context
import android.util.Log
import java.io.File

object ModelProvisioning {
    private const val TAG          = "qwen-asr-prov"
    const val MODEL_DIR_NAME       = "qwen-0.6b"
    const val SAMPLES_DIR_NAME     = "samples"

    fun modelDir(context: Context): File =
        File(context.filesDir, MODEL_DIR_NAME)

    fun samplesDir(context: Context): File =
        File(context.filesDir, SAMPLES_DIR_NAME)

    fun copyModelFromAssetsIfNeeded(context: Context) {
        val dst = modelDir(context)
        val assetSub = "models/$MODEL_DIR_NAME"
        val assetEntries = try {
            context.assets.list(assetSub) ?: emptyArray()
        } catch (_: Exception) { emptyArray() }
        if (assetEntries.isEmpty()) {
            Log.i(TAG, "no bundled model in assets; expecting adb-pushed model at ${dst.absolutePath}")
            return
        }
        if (modelLooksReady(dst, assetEntries)) return
        Log.i(TAG, "copying bundled model to ${dst.absolutePath}")
        copyAssetTree(context, assetSub, dst)
    }

    fun copySamplesFromAssetsIfNeeded(context: Context) {
        val dst = samplesDir(context)
        val assetEntries = try {
            context.assets.list(SAMPLES_DIR_NAME) ?: emptyArray()
        } catch (_: Exception) { emptyArray() }
        if (assetEntries.isEmpty()) return
        if (samplesLookReady(dst, assetEntries)) return
        Log.i(TAG, "copying samples to ${dst.absolutePath}")
        copyAssetTree(context, SAMPLES_DIR_NAME, dst)
    }

    fun modelLooksReady(dir: File, expected: Array<String>): Boolean {
        if (!dir.exists()) return false
        if (expected.isEmpty()) return false
        return expected.all { File(dir, it).exists() }
    }

    fun samplesLookReady(dir: File, expected: Array<String>): Boolean =
        modelLooksReady(dir, expected)

    private fun copyAssetTree(context: Context, assetPath: String, dstRoot: File) {
        val children = context.assets.list(assetPath) ?: emptyArray()
        if (children.isEmpty()) {
            copyAssetFile(context, assetPath, dstRoot)
            return
        }
        if (!dstRoot.exists()) dstRoot.mkdirs()
        for (entry in children) {
            val sub = if (assetPath.isEmpty()) entry else "$assetPath/$entry"
            val subChildren = context.assets.list(sub) ?: emptyArray()
            if (subChildren.isEmpty()) {
                copyAssetFile(context, sub, File(dstRoot, entry))
            } else {
                copyAssetTree(context, sub, File(dstRoot, entry))
            }
        }
    }

    private fun copyAssetFile(context: Context, assetPath: String, dstFile: File) {
        dstFile.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            dstFile.outputStream().use { output -> input.copyTo(output, bufferSize = 64 * 1024) }
        }
    }
}
