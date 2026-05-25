package org.qwen_asr.demo

import android.content.Context
import android.net.Uri
import android.os.Environment
import android.provider.DocumentsContract
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import java.io.File

object UriResolver {
    private const val TAG = "qwen-asr-resolver"

    private const val AUTH_EXTERNAL_STORAGE = "com.android.externalstorage.documents"
    private const val AUTH_DOWNLOADS        = "com.android.providers.downloads.documents"
    private const val RAW_PREFIX            = "raw:"
    private const val PRIMARY_VOLUME        = "primary"

    fun resolveLocalFile(uri: Uri): File? {
        val path = resolveSingleDocumentPath(uri) ?: return null
        val f = File(path)
        return if (f.exists() && f.isFile) f else null
    }

    fun resolveLocalDirectory(context: Context, treeUri: Uri): File? {
        val path = resolveTreeDocumentPath(treeUri) ?: return null
        val dir = File(path)
        if (!dir.exists() || !dir.isDirectory) return null
        if (!containsExpectedSafetensorsLayout(context, treeUri, dir)) return null
        return dir
    }

    private fun resolveSingleDocumentPath(uri: Uri): String? {
        val authority = uri.authority ?: return null
        val docId = runCatching { DocumentsContract.getDocumentId(uri) }.getOrNull() ?: return null
        return decodeDocId(authority, docId)
    }

    private fun resolveTreeDocumentPath(treeUri: Uri): String? {
        val authority = treeUri.authority ?: return null
        val treeDocId = runCatching { DocumentsContract.getTreeDocumentId(treeUri) }.getOrNull()
            ?: return null
        return decodeDocId(authority, treeDocId)
    }

    private fun decodeDocId(authority: String, docId: String): String? {
        return when (authority) {
            AUTH_EXTERNAL_STORAGE -> decodeExternalStorageDocId(docId)
            AUTH_DOWNLOADS        -> decodeDownloadsDocId(docId)
            else                  -> null
        }.also { Log.i(TAG, "decoded authority=$authority docId=$docId -> $it") }
    }

    private fun decodeExternalStorageDocId(docId: String): String? {
        val parts = docId.split(":", limit = 2)
        if (parts.size != 2) return null
        val volume   = parts[0]
        val relative = parts[1]
        if (volume != PRIMARY_VOLUME) return null
        val root = Environment.getExternalStorageDirectory().absolutePath
        return if (relative.isEmpty()) root else "$root/$relative"
    }

    private fun decodeDownloadsDocId(docId: String): String? {
        if (docId.startsWith(RAW_PREFIX)) {
            return docId.removePrefix(RAW_PREFIX)
        }
        return null
    }

    private fun containsExpectedSafetensorsLayout(
        context: Context,
        treeUri: Uri,
        dir:     File,
    ): Boolean {
        if (File(dir, "model.safetensors").exists()) return true
        val root = DocumentFile.fromTreeUri(context, treeUri) ?: return false
        return root.findFile("model.safetensors") != null
    }
}
