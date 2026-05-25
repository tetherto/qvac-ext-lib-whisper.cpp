package org.qwen_asr.demo

import android.content.Context
import android.net.Uri
import android.os.Environment
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream

data class ImportProgress(
    val role:             ModelRole,
    val bytesCopied:      Long,
    val bytesTotal:       Long,
    val currentEntryName: String,
)

data class ImportResult(
    val localPath: String,
    val linked:    Boolean,
)

object ModelImporter {
    private const val TAG       = "qwen-asr-importer"
    private const val LOCAL_DIR = "qwen-models"
    private const val BUFFER    = 256 * 1024

    fun localRootDir(context: Context, role: ModelRole): File =
        File(File(context.filesDir, LOCAL_DIR), role.name.lowercase())

    suspend fun importGgufFile(
        context:     Context,
        sourceUri:   Uri,
        displayName: String,
        onProgress:  suspend (ImportProgress) -> Unit,
    ): ImportResult = withContext(Dispatchers.IO) {
        if (canFopenLocalStorage()) {
            UriResolver.resolveLocalFile(sourceUri)?.let { local ->
                Log.i(TAG, "GGUF linked (no copy): ${local.absolutePath}")
                return@withContext ImportResult(local.absolutePath, linked = true)
            }
        }
        val role     = ModelRole.GGUF
        val destDir  = ensureCleanDir(role, context)
        val destFile = File(destDir, sanitizeFileName(displayName, fallback = "model.gguf"))
        val total    = querySize(context, sourceUri)
        copyUriToFile(
            context     = context,
            sourceUri   = sourceUri,
            destFile    = destFile,
            entryName   = destFile.name,
            role        = role,
            totalBytes  = total,
            baseCopied  = 0L,
            onProgress  = onProgress,
        )
        ImportResult(destFile.absolutePath, linked = false)
    }

    suspend fun importSafetensorsTree(
        context:    Context,
        treeUri:    Uri,
        onProgress: suspend (ImportProgress) -> Unit,
    ): ImportResult = withContext(Dispatchers.IO) {
        if (canFopenLocalStorage()) {
            UriResolver.resolveLocalDirectory(context, treeUri)?.let { local ->
                Log.i(TAG, "safetensors linked (no copy): ${local.absolutePath}")
                return@withContext ImportResult(local.absolutePath, linked = true)
            }
        }
        val role    = ModelRole.SAFETENSORS
        val root    = DocumentFile.fromTreeUri(context, treeUri)
            ?: error("could not open tree uri: $treeUri")
        val destDir = ensureCleanDir(role, context)
        val entries = listFiles(root)
        val total   = entries.sumOf { it.length() }
        var copied  = 0L
        for (entry in entries) {
            val name = entry.name ?: continue
            val destFile = File(destDir, name)
            val uri      = entry.uri
            copied = copyUriToFile(
                context    = context,
                sourceUri  = uri,
                destFile   = destFile,
                entryName  = name,
                role       = role,
                totalBytes = total,
                baseCopied = copied,
                onProgress = onProgress,
            )
        }
        ImportResult(destDir.absolutePath, linked = false)
    }

    fun canFopenLocalStorage(): Boolean =
        runCatching { Environment.isExternalStorageManager() }.getOrDefault(false)

    private fun listFiles(root: DocumentFile): List<DocumentFile> =
        root.listFiles().filter { it.isFile }

    private fun ensureCleanDir(role: ModelRole, context: Context): File {
        val dir = localRootDir(context, role)
        if (dir.exists()) dir.deleteRecursively()
        dir.mkdirs()
        return dir
    }

    private suspend fun copyUriToFile(
        context:    Context,
        sourceUri:  Uri,
        destFile:   File,
        entryName:  String,
        role:       ModelRole,
        totalBytes: Long,
        baseCopied: Long,
        onProgress: suspend (ImportProgress) -> Unit,
    ): Long {
        val input = context.contentResolver.openInputStream(sourceUri)
            ?: error("openInputStream returned null for $sourceUri")
        val written = streamCopy(
            input      = input,
            destFile   = destFile,
            entryName  = entryName,
            role       = role,
            totalBytes = totalBytes,
            baseCopied = baseCopied,
            onProgress = onProgress,
        )
        return baseCopied + written
    }

    private suspend fun streamCopy(
        input:      InputStream,
        destFile:   File,
        entryName:  String,
        role:       ModelRole,
        totalBytes: Long,
        baseCopied: Long,
        onProgress: suspend (ImportProgress) -> Unit,
    ): Long {
        val buffer = ByteArray(BUFFER)
        var written = 0L
        input.use { src ->
            destFile.outputStream().use { dst ->
                while (true) {
                    val read = src.read(buffer)
                    if (read < 0) break
                    dst.write(buffer, 0, read)
                    written += read
                    onProgress(ImportProgress(
                        role             = role,
                        bytesCopied      = baseCopied + written,
                        bytesTotal       = totalBytes,
                        currentEntryName = entryName,
                    ))
                }
            }
        }
        Log.i(TAG, "copied ${destFile.absolutePath} (${written} bytes)")
        return written
    }

    private fun querySize(context: Context, uri: Uri): Long {
        val doc = DocumentFile.fromSingleUri(context, uri)
        return doc?.length() ?: 0L
    }

    private fun sanitizeFileName(name: String, fallback: String): String {
        val cleaned = name.trim().replace("\\s+".toRegex(), "_")
        return cleaned.ifBlank { fallback }
    }
}
