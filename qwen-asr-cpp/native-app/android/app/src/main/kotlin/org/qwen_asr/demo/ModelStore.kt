package org.qwen_asr.demo

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log

enum class ModelRole(val storageKey: String, val displayName: String) {
    SAFETENSORS("model.safetensors.uri.v1", "Safetensors folder"),
    GGUF       ("model.gguf.uri.v1",        "GGUF file"),
}

object ModelStore {
    private const val TAG          = "qwen-asr-store"
    private const val PREFS_FILE   = "qwen-asr-prefs"
    private const val PREFS_SUFFIX_LOCAL = ".local.v1"

    fun savedUri(context: Context, role: ModelRole): Uri? {
        val prefs = context.getSharedPreferences(PREFS_FILE, Context.MODE_PRIVATE)
        return prefs.getString(role.storageKey, null)?.let(Uri::parse)
    }

    fun savedLocalPath(context: Context, role: ModelRole): String? {
        val prefs = context.getSharedPreferences(PREFS_FILE, Context.MODE_PRIVATE)
        return prefs.getString(role.storageKey + PREFS_SUFFIX_LOCAL, null)
    }

    fun persistUri(context: Context, role: ModelRole, uri: Uri, localPath: String) {
        takePersistableRead(context, uri)
        val prefs = context.getSharedPreferences(PREFS_FILE, Context.MODE_PRIVATE)
        prefs.edit()
            .putString(role.storageKey, uri.toString())
            .putString(role.storageKey + PREFS_SUFFIX_LOCAL, localPath)
            .apply()
    }

    fun clear(context: Context, role: ModelRole) {
        savedUri(context, role)?.let { releasePersistableRead(context, it) }
        val prefs = context.getSharedPreferences(PREFS_FILE, Context.MODE_PRIVATE)
        prefs.edit()
            .remove(role.storageKey)
            .remove(role.storageKey + PREFS_SUFFIX_LOCAL)
            .apply()
    }

    private fun takePersistableRead(context: Context, uri: Uri) {
        runCatching {
            context.contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        }.onFailure { Log.w(TAG, "takePersistableUriPermission failed for $uri", it) }
    }

    private fun releasePersistableRead(context: Context, uri: Uri) {
        runCatching {
            context.contentResolver.releasePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        }
    }
}
