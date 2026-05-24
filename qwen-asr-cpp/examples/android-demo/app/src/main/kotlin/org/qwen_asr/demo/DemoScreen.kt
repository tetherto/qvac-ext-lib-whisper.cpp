package org.qwen_asr.demo

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
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

private sealed interface DemoState {
    data object Idle                                : DemoState
    data object LoadingModel                        : DemoState
    data object Transcribing                        : DemoState
    data class  Done(val result: QwenAsrResult)     : DemoState
    data class  Failed(val message: String)         : DemoState
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DemoScreen() {
    val context  = LocalContext.current
    var selected by remember { mutableStateOf(SAMPLES.first()) }
    var state    by remember { mutableStateOf<DemoState>(DemoState.Idle) }
    var engine   by remember { mutableStateOf<QwenAsrEngine?>(null) }
    val scope    = rememberCoroutineScope()

    DisposableEffect(Unit) {
        onDispose { engine?.close() }
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
            LanguageDropdown(selected = selected, onChange = { selected = it; if (state is DemoState.Done || state is DemoState.Failed) state = DemoState.Idle })
            GroundTruthCard(context = context, sample = selected)
            TranscribeButton(state = state) {
                scope.launch {
                    runTranscribe(context, selected, engine, onModelReady = { engine = it }, onState = { state = it })
                }
            }
            when (val s = state) {
                DemoState.Idle           -> Unit
                DemoState.LoadingModel   -> Loading("Loading model (mmap ~1.87 GB safetensors)...")
                DemoState.Transcribing   -> Loading("Transcribing on this device (vendored qwen-asr C kernels)...")
                is DemoState.Failed      -> ErrorCard(s.message)
                is DemoState.Done        -> ResultCard(s.result)
            }
        }
    }
}

@Composable
private fun Header() {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("On-device multilingual ASR demo",
             style = MaterialTheme.typography.titleMedium)
        Text("Backed by the vendored antirez/qwen-asr C kernels. Model: Qwen3-ASR-0.6B (bf16 safetensors, ~1.87 GB).",
             style = MaterialTheme.typography.bodySmall,
             color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    HorizontalDivider()
}

@Composable
private fun LanguageDropdown(selected: LanguageSample, onChange: (LanguageSample) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    Column {
        Text("Language sample",
             style = MaterialTheme.typography.labelMedium,
             color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(4.dp))
        Box {
            ElevatedCard(
                onClick = { expanded = true },
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
private fun TranscribeButton(state: DemoState, onClick: () -> Unit) {
    if (state is DemoState.LoadingModel || state is DemoState.Transcribing) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
    } else {
        Button(onClick = onClick, modifier = Modifier.fillMaxWidth()) {
            androidx.compose.material3.Icon(Icons.Filled.GraphicEq, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Transcribe")
        }
    }
}

@Composable
private fun Loading(message: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator(strokeWidth = 2.dp, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(12.dp))
        Text(message, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun ErrorCard(message: String) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("Error",
             style = MaterialTheme.typography.titleMedium,
             color = MaterialTheme.colorScheme.error)
        Box(modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0xCC, 0x33, 0x33, 0x22))
            .fillMaxWidth()
            .padding(12.dp)) {
            Text(message,
                 style = MaterialTheme.typography.bodySmall,
                 fontFamily = FontFamily.Monospace)
        }
    }
}

@Composable
private fun ResultCard(r: QwenAsrResult) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Transcript", style = MaterialTheme.typography.titleMedium)
        Box(modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0x22, 0x99, 0x55, 0x33))
            .fillMaxWidth()
            .padding(12.dp)) {
            Text(r.text, style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Serif)
        }

        Text("Performance", style = MaterialTheme.typography.titleMedium)
        Box(modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0x99, 0x99, 0x99, 0x22))
            .fillMaxWidth()
            .padding(12.dp)) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                metricRow("Tokens",  "${r.tokens}")
                metricRow("Audio",   "%.0f ms".format(r.audioMs))
                metricRow("Encode",  "%.0f ms".format(r.encodeMs))
                metricRow("Decode",  "%.0f ms".format(r.decodeMs))
                metricRow("Total",   "%.0f ms".format(r.totalMs))
                Row {
                    Text("RTFx",
                         modifier = Modifier.width(80.dp),
                         color = MaterialTheme.colorScheme.onSurfaceVariant,
                         fontFamily = FontFamily.Monospace,
                         fontSize = 12.sp)
                    Text("%.2fx realtime".format(r.realTimeFactor),
                         color = Color(0x00, 0x88, 0x44, 0xFF),
                         fontWeight = FontWeight.Bold,
                         fontFamily = FontFamily.Monospace,
                         fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
private fun metricRow(k: String, v: String) {
    Row {
        Text(k,
             modifier = Modifier.width(80.dp),
             color = MaterialTheme.colorScheme.onSurfaceVariant,
             fontFamily = FontFamily.Monospace,
             fontSize = 12.sp)
        Text(v, fontFamily = FontFamily.Monospace, fontSize = 12.sp)
    }
}

private suspend fun runTranscribe(
    context: Context,
    sample:  LanguageSample,
    initial: QwenAsrEngine?,
    onModelReady: (QwenAsrEngine) -> Unit,
    onState: (DemoState) -> Unit,
) {
    val wavFile = File(ModelProvisioning.samplesDir(context), "${sample.basename}.wav")
    if (!wavFile.exists()) {
        onState(DemoState.Failed("sample not found on device: ${wavFile.absolutePath}"))
        return
    }
    val modelDir = ModelProvisioning.modelDir(context)
    if (!File(modelDir, "model.safetensors").exists()) {
        onState(DemoState.Failed("model not provisioned at ${modelDir.absolutePath}\n\nRun: bash scripts/push-model-android.sh"))
        return
    }

    try {
        var engine = initial
        if (engine == null) {
            onState(DemoState.LoadingModel)
            engine = withContext(Dispatchers.Default) { QwenAsrEngine(modelDir.absolutePath, nThreads = INFERENCE_THREADS) }
            onModelReady(engine)
        }
        onState(DemoState.Transcribing)
        val r = withContext(Dispatchers.Default) { engine.transcribe(wavFile.absolutePath) }
        onState(DemoState.Done(r))
    } catch (t: Throwable) {
        onState(DemoState.Failed(t.message ?: t.toString()))
    }
}

private fun readGroundTruth(context: Context, sample: LanguageSample): String {
    val f = File(ModelProvisioning.samplesDir(context), "${sample.basename}.txt")
    return runCatching { f.readText().trim() }.getOrDefault("")
}
