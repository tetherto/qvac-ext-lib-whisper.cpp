'use strict'

function formatNumber (value, digits = 3) {
  if (value == null || Number.isNaN(value)) return 'n/a'
  return Number(value).toFixed(digits)
}

function formatBytes (value) {
  if (value == null || Number.isNaN(value)) return 'n/a'
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`
}

function renderEngineTable (title, engines) {
  const lines = [
    `## ${title}`,
    '',
    '| Engine | Success | Fail | Median gen ms | Median e2e ms | Median RTF | Peak RSS | Unique WAV hashes | Silent |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|'
  ]
  for (const [name, stats] of Object.entries(engines || {})) {
    lines.push(`| ${name} | ${stats.successCount} | ${stats.failureCount} | ${formatNumber(stats.generationMs.median, 1)} | ${formatNumber(stats.e2eMs.median, 1)} | ${formatNumber(stats.rtf.median, 3)} | ${formatBytes(stats.peakRssBytes.median)} | ${stats.uniqueWavHashes} | ${stats.silentCount} |`)
  }
  lines.push('')
  return lines
}

function renderMarkdownReport (data) {
  const lines = [
    `# ACE-Step engine comparison (${data.meta.backend})`,
    '',
    `Generated: ${data.meta.generatedAt}`,
    '',
    `Comparison class: **${data.meta.comparisonClass}**. Platform: \`${data.meta.platform}\`. Threads: ${data.meta.threads}. Warm-ups: ${data.meta.warmups}. Timed runs: ${data.meta.runs}. Order: \`${data.meta.order}\`. Cooldown: ${data.meta.cooldownMs} ms.`,
    '',
    'Both engines used the same GGUF files, prompt manifest, duration, seed, LM sampling defaults, Euler sampler, and Haar DCW double-mode scalers. QVAC `music-cli` is a single process. `acestep.cpp` is `ace-lm` then `ace-synth`. QVAC lazy-loads stages inside `generate()`; acestep.cpp loads modules in each CLI process.',
    '',
    ...renderEngineTable('Overall', data.overall)
  ]
  for (const [promptId, engines] of Object.entries(data.byPrompt || {})) {
    lines.push(...renderEngineTable(promptId, engines))
  }
  lines.push('Failed rounds are retained in the JSON. Quality metrics are computed from WAV files and are not included in generation time.')
  lines.push('')
  return lines.join('\n')
}

module.exports = {
  formatBytes,
  formatNumber,
  renderMarkdownReport
}
