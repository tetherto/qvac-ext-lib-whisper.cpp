package org.qwen_asr.demo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.tooling.preview.Preview

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ModelProvisioning.copyModelFromAssetsIfNeeded(this)
        ModelProvisioning.copySamplesFromAssetsIfNeeded(this)
        setContent { AppRoot() }
    }
}

@Composable
private fun AppRoot() {
    MaterialTheme {
        Surface(color = MaterialTheme.colorScheme.background) {
            DemoScreen()
        }
    }
}

@Preview(showBackground = true)
@Composable
fun AppRootPreview() {
    AppRoot()
}
