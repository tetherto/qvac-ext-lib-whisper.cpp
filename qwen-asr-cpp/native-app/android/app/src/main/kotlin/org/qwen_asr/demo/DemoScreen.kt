package org.qwen_asr.demo

import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.HourglassEmpty
import androidx.compose.material.icons.filled.InsertDriveFile
import androidx.compose.material.icons.filled.Inventory2
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

private const val INFERENCE_THREADS = 8

private data class LanguageSample(
    val id:       String,
    val label:    String,
    val flag:     String,
    val basename: String,
)

private val SAMPLES = listOf(
    LanguageSample("en", "English (JFK)",       "US", "jfk"),
    LanguageSample("es", "Spanish (LatAm)",     "MX", "es"),
    LanguageSample("it", "Italian",             "IT", "it"),
    LanguageSample("ru", "Russian",             "RU", "ru"),
    LanguageSample("ar", "Arabic (Egyptian)",   "EG", "ar"),
    LanguageSample("zh", "Chinese (Mandarin)",  "CN", "zh"),
    LanguageSample("ja", "Japanese",            "JP", "ja"),
    LanguageSample("ko", "Korean",              "KR", "ko"),
    LanguageSample("hi", "Hindi",               "IN", "hi"),
)

private sealed interface BackendState {
    data object Idle                                : BackendState
    data object LoadingModel                        : BackendState
    data object Transcribing                        : BackendState
    data class  Done(val result: QwenAsrResult)     : BackendState
    data class  Failed(val message: String)         : BackendState
}

private sealed interface ImportState {
    data object Idle                                 : ImportState
    data class  Working(val progress: ImportProgress): ImportState
    data class  Failed(val message: String)          : ImportState
}

private class SamplePlayer {
    private var mediaPlayer: MediaPlayer? = null
    var isPlaying by mutableStateOf(false)
        private set

    fun toggle(file: File, onComplete: () -> Unit = {}) {
        if (isPlaying) {
            stop()
            return
        }
        if (!file.exists()) return
        val mp = MediaPlayer().apply {
            setDataSource(file.absolutePath)
            setOnCompletionListener {
                this@SamplePlayer.isPlaying = false
                onComplete()
            }
            setOnErrorListener { _, _, _ -> this@SamplePlayer.isPlaying = false; onComplete(); true }
            prepare()
            start()
        }
        mediaPlayer = mp
        isPlaying   = true
    }

    fun stop() {
        mediaPlayer?.let {
            runCatching { it.stop() }
            it.release()
        }
        mediaPlayer = null
        isPlaying   = false
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DemoScreen() {
    val context  = LocalContext.current
    var selected by remember { mutableStateOf(SAMPLES.first()) }
    val perBackend = remember {
        mutableStateMapOf<QwenAsrBackend, BackendState>().apply {
            QwenAsrBackend.entries.forEach { put(it, BackendState.Idle) }
        }
    }
    val engines = remember { mutableStateMapOf<QwenAsrBackend, QwenAsrEngine>() }
    var isRunning by remember { mutableStateOf(false) }
    var currentlyRunning by remember { mutableStateOf<QwenAsrBackend?>(null) }
    var hasRunOnce by remember { mutableStateOf(false) }
    val player    = remember { SamplePlayer() }
    val scope     = rememberCoroutineScope()

    val importStates = remember {
        mutableStateMapOf<ModelRole, ImportState>().apply {
            ModelRole.entries.forEach { put(it, ImportState.Idle) }
        }
    }
    val pathTicks = remember {
        mutableStateMapOf<ModelRole, Int>().apply {
            ModelRole.entries.forEach { put(it, 0) }
        }
    }
    var allFilesAccess by remember { mutableStateOf(hasAllFilesAccess()) }
    val allFilesLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { allFilesAccess = hasAllFilesAccess() }

    DisposableEffect(Unit) {
        onDispose {
            engines.values.forEach { it.close() }
            engines.clear()
            player.stop()
        }
    }

    Scaffold(topBar = {
        CenterAlignedTopAppBar(title = { Text("Qwen3-ASR · 0.6B") })
    }) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Header()
            AllFilesAccessBanner(
                granted      = allFilesAccess,
                onGrantClick = { openAllFilesAccessSettings(context, allFilesLauncher) },
            )
            ModelFilesSection(
                context        = context,
                importStates   = importStates,
                pathTicks      = pathTicks,
                onPicked       = { role, uri, displayName ->
                    scope.launch {
                        importModel(
                            context      = context,
                            role         = role,
                            uri          = uri,
                            displayName  = displayName,
                            importStates = importStates,
                            pathTicks    = pathTicks,
                        )
                    }
                },
                onClear        = { role ->
                    ModelImporter.localRootDir(context, role).deleteRecursively()
                    ModelStore.clear(context, role)
                    importStates[role] = ImportState.Idle
                    pathTicks[role]    = (pathTicks[role] ?: 0) + 1
                },
            )
            HorizontalDivider()
            LanguageDropdown(
                context  = context,
                selected = selected,
                player   = player,
                onChange = {
                    if (selected != it) player.stop()
                    selected = it
                    QwenAsrBackend.entries.forEach { b -> perBackend[b] = BackendState.Idle }
                })
            GroundTruthCard(context = context, sample = selected)
            RunButtons(
                context          = context,
                pathTicks        = pathTicks,
                isRunning        = isRunning,
                currentlyRunning = currentlyRunning,
                onRun            = { backend ->
                    scope.launch {
                        isRunning        = true
                        currentlyRunning = backend
                        hasRunOnce       = true
                        runSingleBackend(
                            context     = context,
                            backend     = backend,
                            sample      = selected,
                            engines     = engines,
                            perBackend  = perBackend,
                        )
                        isRunning        = false
                        currentlyRunning = null
                    }
                })
            if (hasRunOnce) {
                ResultsSection(perBackend = perBackend)
            }
        }
    }
}

@Composable
private fun Header() {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("On-device multilingual ASR · backend A/B comparison",
             style = MaterialTheme.typography.titleMedium)
        Text("Safetensors (v0.1, vendored antirez/qwen-asr) and GGUF (v0.2, our GGML port) run separately so peak RAM stays low on phones.",
             style = MaterialTheme.typography.bodySmall,
             color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    HorizontalDivider()
}

@Composable
private fun ModelFilesSection(
    context:      Context,
    importStates: Map<ModelRole, ImportState>,
    pathTicks:    Map<ModelRole, Int>,
    onPicked:     (ModelRole, Uri, String) -> Unit,
    onClear:      (ModelRole) -> Unit,
) {
    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0x99, 0x99, 0x99, 0x14))
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text("Step 1 · Model files on device",
             style = MaterialTheme.typography.titleSmall,
             fontWeight = FontWeight.Bold)
        Text("Pick the safetensors folder and the GGUF file from your device storage (Files, Drive, Downloads...). The app copies them once into its private sandbox so the C++ engine can mmap them.",
             style = MaterialTheme.typography.labelSmall,
             color = MaterialTheme.colorScheme.onSurfaceVariant)
        ModelPickerRow(
            context        = context,
            role           = ModelRole.SAFETENSORS,
            importState    = importStates[ModelRole.SAFETENSORS] ?: ImportState.Idle,
            pathTick       = pathTicks[ModelRole.SAFETENSORS] ?: 0,
            onPicked       = onPicked,
            onClear        = onClear,
        )
        ModelPickerRow(
            context        = context,
            role           = ModelRole.GGUF,
            importState    = importStates[ModelRole.GGUF] ?: ImportState.Idle,
            pathTick       = pathTicks[ModelRole.GGUF] ?: 0,
            onPicked       = onPicked,
            onClear        = onClear,
        )
    }
}

@Composable
private fun ModelPickerRow(
    context:     Context,
    role:        ModelRole,
    importState: ImportState,
    pathTick:    Int,
    onPicked:    (ModelRole, Uri, String) -> Unit,
    onClear:     (ModelRole) -> Unit,
) {
    val backend = backendFor(role)
    val accent  = backendColor(backend)
    val localPath = remember(pathTick) { ModelProvisioning.modelPath(context, backend) }
    val isWorking = importState is ImportState.Working

    val folderLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        if (uri != null) onPicked(role, uri, uri.lastPathSegment ?: "safetensors")
    }
    val fileLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) onPicked(role, uri, displayNameFromUri(context, uri) ?: "qwen3-asr.gguf")
    }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            androidx.compose.material3.Icon(
                imageVector        = backendIcon(backend),
                contentDescription = null,
                tint               = accent
            )
            Spacer(Modifier.width(6.dp))
            Text(role.displayName,
                 style       = MaterialTheme.typography.bodyMedium,
                 fontWeight  = FontWeight.Bold,
                 modifier    = Modifier.weight(1f))
            ModelStatusBadge(localPath = localPath, importState = importState, accent = accent)
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(
                onClick  = {
                    when (role) {
                        ModelRole.SAFETENSORS -> folderLauncher.launch(null)
                        ModelRole.GGUF        -> fileLauncher.launch(arrayOf("*/*"))
                    }
                },
                enabled  = !isWorking,
                modifier = Modifier.weight(1f),
            ) {
                androidx.compose.material3.Icon(
                    imageVector = when (role) {
                        ModelRole.SAFETENSORS -> Icons.Filled.Folder
                        ModelRole.GGUF        -> Icons.Filled.InsertDriveFile
                    },
                    contentDescription = null
                )
                Spacer(Modifier.width(6.dp))
                Text(if (localPath != null) "Change" else "Select")
            }
            if (localPath != null) {
                Spacer(Modifier.width(8.dp))
                TextButton(onClick = { onClear(role) }, enabled = !isWorking) {
                    Text("Clear", color = MaterialTheme.colorScheme.error)
                }
            }
        }
        if (importState is ImportState.Working) {
            ImportProgressView(importState.progress)
        }
        if (importState is ImportState.Failed) {
            Text(importState.message,
                 style = MaterialTheme.typography.labelSmall,
                 color = MaterialTheme.colorScheme.error,
                 fontFamily = FontFamily.Monospace)
        }
        if (localPath != null && importState !is ImportState.Working) {
            Text(localPath,
                 style = MaterialTheme.typography.labelSmall,
                 color = MaterialTheme.colorScheme.onSurfaceVariant,
                 fontFamily = FontFamily.Monospace,
                 fontSize = 10.sp)
        }
    }
}

@Composable
private fun ModelStatusBadge(localPath: String?, importState: ImportState, accent: Color) {
    val (text, color, icon) = when {
        importState is ImportState.Working -> Triple("copying...",    accent,                                  Icons.Filled.HourglassEmpty)
        importState is ImportState.Failed  -> Triple("failed",        MaterialTheme.colorScheme.error,        Icons.Filled.Warning)
        localPath != null                  -> Triple("ready",         accent,                                  Icons.Filled.CheckCircle)
        else                               -> Triple("not selected",  Color(0xCC, 0x88, 0x00, 0xFF),          Icons.Filled.Warning)
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        androidx.compose.material3.Icon(
            imageVector        = icon,
            contentDescription = null,
            tint               = color,
            modifier           = Modifier.size(16.dp)
        )
        Spacer(Modifier.width(4.dp))
        Text(text, style = MaterialTheme.typography.labelSmall, color = color)
    }
}

@Composable
private fun ImportProgressView(progress: ImportProgress) {
    val fraction = if (progress.bytesTotal > 0) {
        (progress.bytesCopied.toFloat() / progress.bytesTotal.toFloat()).coerceIn(0f, 1f)
    } else 0f
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
        Text(
            text = "${humanBytes(progress.bytesCopied)} / ${humanBytes(progress.bytesTotal)} · ${progress.currentEntryName}",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontFamily = FontFamily.Monospace,
            fontSize = 10.sp,
        )
    }
}

private fun humanBytes(b: Long): String {
    if (b < 1024) return "${b} B"
    val kib = b / 1024.0
    if (kib < 1024) return "%.0f KB".format(kib)
    val mib = kib / 1024.0
    if (mib < 1024) return "%.1f MB".format(mib)
    val gib = mib / 1024.0
    return "%.2f GB".format(gib)
}

private fun displayNameFromUri(context: Context, uri: Uri): String? {
    return runCatching {
        val doc = androidx.documentfile.provider.DocumentFile.fromSingleUri(context, uri)
        doc?.name
    }.getOrNull()
}

private fun backendFor(role: ModelRole): QwenAsrBackend = when (role) {
    ModelRole.SAFETENSORS -> QwenAsrBackend.SAFETENSORS
    ModelRole.GGUF        -> QwenAsrBackend.GGUF
}

@Composable
private fun LanguageDropdown(
    context:  Context,
    selected: LanguageSample,
    player:   SamplePlayer,
    onChange: (LanguageSample) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    val wavFile = remember(selected) { File(ModelProvisioning.samplesDir(context), "${selected.basename}.wav") }
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            androidx.compose.material3.Icon(
                imageVector = Icons.Filled.GraphicEq,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.width(6.dp))
            Text("Step 2 · Audio sample",
                 style = MaterialTheme.typography.titleSmall,
                 fontWeight = FontWeight.Bold)
        }
        Spacer(Modifier.height(4.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(modifier = Modifier.weight(1f)) {
                ElevatedCard(
                    onClick  = { expanded = true },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 12.dp).fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("${selected.flag}  ${selected.label}",
                             style = MaterialTheme.typography.bodyMedium)
                        Spacer(Modifier.weight(1f))
                        androidx.compose.material3.Icon(
                            imageVector = Icons.Filled.ExpandMore,
                            contentDescription = null
                        )
                    }
                }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    for (s in SAMPLES) {
                        DropdownMenuItem(
                            text = { Text("${s.flag}  ${s.label}") },
                            onClick = { onChange(s); expanded = false }
                        )
                    }
                }
            }
            Spacer(Modifier.width(8.dp))
            IconButton(
                onClick = { player.toggle(wavFile) },
                enabled = wavFile.exists()
            ) {
                androidx.compose.material3.Icon(
                    imageVector = if (player.isPlaying) Icons.Filled.Stop else Icons.Filled.PlayArrow,
                    contentDescription = if (player.isPlaying) "Stop preview" else "Play preview"
                )
            }
        }
        Spacer(Modifier.height(4.dp))
        Text(
            text  = if (wavFile.exists()) wavFile.name else "${selected.basename}.wav (not on device)",
            style = MaterialTheme.typography.labelSmall,
            color = if (wavFile.exists()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error
        )
    }
}

@Composable
private fun GroundTruthCard(context: Context, sample: LanguageSample) {
    val truth = remember(sample) { readGroundTruth(context, sample) }
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("Ground truth (FLEURS)",
             style = MaterialTheme.typography.labelSmall,
             color = MaterialTheme.colorScheme.onSurfaceVariant)
        Box(modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0x33, 0x66, 0xCC, 0x33))
            .fillMaxWidth()
            .padding(12.dp)) {
            Text(if (truth.isBlank()) "— no reference available —" else truth,
                 style = MaterialTheme.typography.bodySmall,
                 fontFamily = FontFamily.Serif)
        }
    }
}

@Composable
private fun RunButtons(
    context:          Context,
    pathTicks:        Map<ModelRole, Int>,
    isRunning:        Boolean,
    currentlyRunning: QwenAsrBackend?,
    onRun:            (QwenAsrBackend) -> Unit,
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for (backend in QwenAsrBackend.entries) {
            Box(Modifier.weight(1f)) {
                BackendRunButton(
                    context          = context,
                    backend          = backend,
                    tick             = pathTicks[roleFor(backend)] ?: 0,
                    isRunning        = isRunning,
                    currentlyRunning = currentlyRunning,
                    onRun            = onRun,
                )
            }
        }
    }
}

@Composable
private fun BackendRunButton(
    context:          Context,
    backend:          QwenAsrBackend,
    tick:             Int,
    isRunning:        Boolean,
    currentlyRunning: QwenAsrBackend?,
    onRun:            (QwenAsrBackend) -> Unit,
) {
    val present = remember(backend, tick, isRunning) { ModelProvisioning.modelPath(context, backend) != null }
    val isThisRunning = isRunning && currentlyRunning == backend
    val label = when {
        isThisRunning -> "${shortLabel(backend)}..."
        else          -> "Run ${shortLabel(backend)}"
    }
    Button(
        onClick  = { onRun(backend) },
        enabled  = present && !isRunning,
        colors   = ButtonDefaults.buttonColors(containerColor = backendColor(backend)),
        modifier = Modifier.fillMaxWidth()
    ) {
        androidx.compose.material3.Icon(
            imageVector        = if (isThisRunning) Icons.Filled.HourglassEmpty else backendIcon(backend),
            contentDescription = null
        )
        Spacer(Modifier.width(6.dp))
        Text(label)
    }
}

private fun roleFor(backend: QwenAsrBackend): ModelRole = when (backend) {
    QwenAsrBackend.SAFETENSORS -> ModelRole.SAFETENSORS
    QwenAsrBackend.GGUF        -> ModelRole.GGUF
}

private fun shortLabel(backend: QwenAsrBackend) = when (backend) {
    QwenAsrBackend.SAFETENSORS -> "Safetensors"
    QwenAsrBackend.GGUF        -> "GGUF"
}

@Composable
private fun ResultsSection(perBackend: Map<QwenAsrBackend, BackendState>) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            androidx.compose.material3.Icon(
                imageVector = Icons.Filled.GraphicEq,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.width(6.dp))
            Text("Step 3 · Results · side by side",
                 style = MaterialTheme.typography.titleSmall,
                 fontWeight = FontWeight.Bold)
        }
        BackendsGrid(perBackend)
    }
}

@Composable
private fun BackendsGrid(perBackend: Map<QwenAsrBackend, BackendState>) {
    BoxWithConstraints {
        val sideBySide = maxWidth > 600.dp
        if (sideBySide) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                for (b in QwenAsrBackend.entries) {
                    Box(Modifier.weight(1f)) {
                        BackendCard(b, perBackend[b] ?: BackendState.Idle)
                    }
                }
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                for (b in QwenAsrBackend.entries) {
                    BackendCard(b, perBackend[b] ?: BackendState.Idle)
                }
            }
        }
    }
}

@Composable
private fun BackendCard(b: QwenAsrBackend, state: BackendState) {
    val accent = backendColor(b)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(Color(0x99, 0x99, 0x99, 0x14))
            .border(width = 1.dp, color = accent.copy(alpha = 0.25f), shape = RoundedCornerShape(10.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            androidx.compose.material3.Icon(
                imageVector = backendIcon(b),
                contentDescription = null,
                tint = accent
            )
            Spacer(Modifier.width(6.dp))
            Text(b.displayName,
                 style = MaterialTheme.typography.titleSmall,
                 fontWeight = FontWeight.Bold)
            Spacer(Modifier.weight(1f))
            StatusBadge(state)
        }
        StateContent(b, state)
    }
}

@Composable
private fun StatusBadge(state: BackendState) {
    val (text, color) = when (state) {
        is BackendState.Idle           -> "idle"     to MaterialTheme.colorScheme.onSurfaceVariant
        is BackendState.LoadingModel   -> "loading"  to Color(0xCC, 0x88, 0x00, 0xFF)
        is BackendState.Transcribing   -> "running"  to Color(0xCC, 0x88, 0x00, 0xFF)
        is BackendState.Done           -> "done"     to Color(0x00, 0x88, 0x44, 0xFF)
        is BackendState.Failed         -> "failed"   to MaterialTheme.colorScheme.error
    }
    Text(text, fontSize = 11.sp, color = color)
}

@Composable
private fun StateContent(b: QwenAsrBackend, state: BackendState) {
    when (state) {
        is BackendState.Idle -> Text("Idle.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)

        is BackendState.LoadingModel -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(strokeWidth = 2.dp, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(8.dp))
            Text(loadingMessage(b), style = MaterialTheme.typography.bodySmall)
        }

        is BackendState.Transcribing -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(strokeWidth = 2.dp, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(8.dp))
            Text(transcribingMessage(b), style = MaterialTheme.typography.bodySmall)
        }

        is BackendState.Failed -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Error", style = MaterialTheme.typography.labelSmall,
                 color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold)
            Box(modifier = Modifier
                .clip(RoundedCornerShape(6.dp))
                .background(Color(0xCC, 0x33, 0x33, 0x22))
                .fillMaxWidth()
                .padding(8.dp)) {
                Text(state.message,
                     style = MaterialTheme.typography.labelSmall,
                     fontFamily = FontFamily.Monospace)
            }
        }

        is BackendState.Done -> Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Box(modifier = Modifier
                .clip(RoundedCornerShape(6.dp))
                .background(Color(0x22, 0x99, 0x55, 0x33))
                .fillMaxWidth()
                .padding(8.dp)) {
                Text(state.result.text,
                     style = MaterialTheme.typography.bodySmall,
                     fontFamily = FontFamily.Serif)
            }
            CompactMetrics(state.result)
        }
    }
}

@Composable
private fun CompactMetrics(r: QwenAsrResult) {
    Box(modifier = Modifier
        .clip(RoundedCornerShape(6.dp))
        .background(Color(0x99, 0x99, 0x99, 0x18))
        .fillMaxWidth()
        .padding(8.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            metricRow("tokens", "${r.tokens}")
            metricRow("audio",  "%.0f ms".format(r.audioMs))
            metricRow("encode", "%.0f ms".format(r.encodeMs))
            metricRow("decode", "%.0f ms".format(r.decodeMs))
            metricRow("total",  "%.0f ms".format(r.totalMs))
            Row {
                Text("rtfx",
                     modifier = Modifier.width(64.dp),
                     color = MaterialTheme.colorScheme.onSurfaceVariant,
                     fontFamily = FontFamily.Monospace,
                     fontSize = 11.sp)
                Text("%.2fx".format(r.realTimeFactor),
                     color = Color(0x00, 0x88, 0x44, 0xFF),
                     fontWeight = FontWeight.Bold,
                     fontFamily = FontFamily.Monospace,
                     fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun metricRow(k: String, v: String) {
    Row {
        Text(k,
             modifier = Modifier.width(64.dp),
             color = MaterialTheme.colorScheme.onSurfaceVariant,
             fontFamily = FontFamily.Monospace,
             fontSize = 11.sp)
        Text(v, fontFamily = FontFamily.Monospace, fontSize = 11.sp)
    }
}

private fun backendIcon(b: QwenAsrBackend) = when (b) {
    QwenAsrBackend.SAFETENSORS -> Icons.Filled.Inventory2
    QwenAsrBackend.GGUF        -> Icons.Filled.Bolt
}

private fun backendColor(b: QwenAsrBackend) = when (b) {
    QwenAsrBackend.SAFETENSORS -> Color(0x33, 0x66, 0xCC, 0xFF)
    QwenAsrBackend.GGUF        -> Color(0x19, 0x73, 0x33, 0xFF)
}

private fun loadingMessage(b: QwenAsrBackend): String = when (b) {
    QwenAsrBackend.SAFETENSORS -> "Loading 1.87 GB safetensors (mmap)..."
    QwenAsrBackend.GGUF        -> "Loading GGUF (GGML runtime)..."
}

private fun transcribingMessage(b: QwenAsrBackend): String = when (b) {
    QwenAsrBackend.SAFETENSORS -> "Transcribing (vendored qwen-asr C kernels)..."
    QwenAsrBackend.GGUF        -> "Transcribing (GGML)..."
}

private suspend fun importModel(
    context:      Context,
    role:         ModelRole,
    uri:          Uri,
    displayName:  String,
    importStates: MutableMap<ModelRole, ImportState>,
    pathTicks:    MutableMap<ModelRole, Int>,
) {
    importStates[role] = ImportState.Working(ImportProgress(role, 0, 0, displayName))
    try {
        val result = when (role) {
            ModelRole.SAFETENSORS -> ModelImporter.importSafetensorsTree(
                context     = context,
                treeUri     = uri,
                onProgress  = { p ->
                    withContext(Dispatchers.Main) { importStates[role] = ImportState.Working(p) }
                },
            )
            ModelRole.GGUF -> ModelImporter.importGgufFile(
                context     = context,
                sourceUri   = uri,
                displayName = displayName,
                onProgress  = { p ->
                    withContext(Dispatchers.Main) { importStates[role] = ImportState.Working(p) }
                },
            )
        }
        ModelStore.persistUri(context, role, uri, result.localPath)
        importStates[role] = ImportState.Idle
        pathTicks[role]    = (pathTicks[role] ?: 0) + 1
    } catch (t: Throwable) {
        importStates[role] = ImportState.Failed(t.message ?: t.toString())
    }
}

private fun hasAllFilesAccess(): Boolean =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) Environment.isExternalStorageManager() else true

private fun openAllFilesAccessSettings(
    context:  Context,
    launcher: androidx.activity.result.ActivityResultLauncher<Intent>,
) {
    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
        data = Uri.parse("package:${context.packageName}")
    }
    runCatching { launcher.launch(intent) }.onFailure {
        launcher.launch(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
    }
}

@Composable
private fun AllFilesAccessBanner(granted: Boolean, onGrantClick: () -> Unit) {
    val accent = if (granted) Color(0x19, 0x73, 0x33, 0xFF) else Color(0xCC, 0x88, 0x00, 0xFF)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(accent.copy(alpha = 0.10f))
            .border(1.dp, accent.copy(alpha = 0.30f), RoundedCornerShape(8.dp))
            .padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        androidx.compose.material3.Icon(
            imageVector        = if (granted) Icons.Filled.CheckCircle else Icons.Filled.Warning,
            contentDescription = null,
            tint               = accent
        )
        Spacer(Modifier.width(8.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                if (granted) "All Files Access granted — models will be linked (no copy)"
                else         "Optional: grant All Files Access to skip copying models",
                style      = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color      = accent,
            )
            Text(
                if (granted) "Selected files in /sdcard/Download/... will be read in place."
                else         "Without it, picked models are copied into the app sandbox.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (!granted) {
            Spacer(Modifier.width(8.dp))
            TextButton(onClick = onGrantClick) { Text("Grant") }
        }
    }
}

private suspend fun runSingleBackend(
    context:    Context,
    backend:    QwenAsrBackend,
    sample:     LanguageSample,
    engines:    MutableMap<QwenAsrBackend, QwenAsrEngine>,
    perBackend: MutableMap<QwenAsrBackend, BackendState>,
) {
    val wavFile = File(ModelProvisioning.samplesDir(context), "${sample.basename}.wav")
    if (!wavFile.exists()) {
        withContext(Dispatchers.Main) {
            perBackend[backend] = BackendState.Failed("sample not on device: ${wavFile.absolutePath}")
        }
        return
    }
    val modelPath = ModelProvisioning.modelPath(context, backend)
    if (modelPath == null) {
        withContext(Dispatchers.Main) {
            perBackend[backend] = BackendState.Failed(
                "no model selected for ${backend.displayName}")
        }
        return
    }
    val wavPath = wavFile.absolutePath
    try {
        runInferenceLifecycle(backend, modelPath, wavPath, engines, perBackend)
    } catch (t: Throwable) {
        withContext(Dispatchers.Main) {
            perBackend[backend] = BackendState.Failed(t.message ?: t.toString())
        }
    }
}

private suspend fun runInferenceLifecycle(
    backend:    QwenAsrBackend,
    modelPath:  String,
    wavPath:    String,
    engines:    MutableMap<QwenAsrBackend, QwenAsrEngine>,
    perBackend: MutableMap<QwenAsrBackend, BackendState>,
) {
    withContext(Dispatchers.Main) { perBackend[backend] = BackendState.LoadingModel }
    val engine = withContext(Dispatchers.Default) {
        QwenAsrEngine(modelPath, backend = backend, nThreads = INFERENCE_THREADS)
    }
    engines[backend] = engine
    withContext(Dispatchers.Main) { perBackend[backend] = BackendState.Transcribing }
    val result = withContext(Dispatchers.Default) { engine.transcribe(wavPath) }
    withContext(Dispatchers.Main) { perBackend[backend] = BackendState.Done(result) }
    releaseEngine(backend, engines)
}

private suspend fun releaseEngine(
    backend: QwenAsrBackend,
    engines: MutableMap<QwenAsrBackend, QwenAsrEngine>,
) {
    withContext(Dispatchers.Default) {
        engines.remove(backend)?.close()
        System.gc()
    }
}

private fun readGroundTruth(context: Context, sample: LanguageSample): String {
    val f = File(ModelProvisioning.samplesDir(context), "${sample.basename}.txt")
    return runCatching { f.readText().trim() }.getOrDefault("")
}
