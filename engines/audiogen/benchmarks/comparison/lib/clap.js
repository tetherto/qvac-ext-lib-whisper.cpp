'use strict'

function buildClapText (prompt, textPolicy) {
  if (!prompt || typeof prompt.caption !== 'string') {
    throw new Error('CLAP prompt must include a caption string')
  }
  if (textPolicy === 'caption+lyrics') {
    const lyrics = prompt.lyrics == null ? '' : String(prompt.lyrics)
    return `${prompt.caption}\n${lyrics}`
  }
  if (textPolicy && textPolicy !== 'caption') {
    throw new Error(`unknown CLAP text policy: ${textPolicy}`)
  }
  return prompt.caption
}

function parseClapBatchOutput (stdout) {
  const text = String(stdout || '').trim()
  if (!text) throw new Error('CLAP scorer printed no JSON')
  let parsed
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw new Error(`CLAP scorer stdout is not JSON: ${error.message}`)
  }
  if (!parsed || parsed.ok !== true || !Array.isArray(parsed.scores)) {
    const detail = parsed && parsed.error ? parsed.error : 'missing scores array'
    throw new Error(`CLAP scorer failed: ${detail}`)
  }
  return parsed
}

function shouldScoreRound (record, options) {
  const timedOnly = options.timedOnly !== false
  if (!record) return false
  if (timedOnly && record.kind !== 'timed') return false
  if (!record.wavPath) return false
  if (!options.force && record.clap && typeof record.clap.score === 'number') return false
  return true
}

function mergeClapScore (record, scoreItem, meta) {
  const clap = {
    ok: Boolean(scoreItem && scoreItem.ok),
    score: scoreItem && typeof scoreItem.score === 'number' ? scoreItem.score : null,
    error: scoreItem && scoreItem.error ? scoreItem.error : null,
    elapsedMs: scoreItem && scoreItem.elapsedMs != null ? scoreItem.elapsedMs : null,
    model: meta.model,
    revision: meta.revision || null,
    samplingRate: meta.samplingRate || null,
    device: meta.device,
    textPolicy: meta.textPolicy
  }
  return { ...record, clap }
}

function lookupPrompt (prompts, promptId) {
  const prompt = prompts.find(entry => entry.id === promptId)
  if (!prompt) throw new Error(`CLAP prompt id not in manifest: ${promptId}`)
  return prompt
}

module.exports = {
  buildClapText,
  lookupPrompt,
  mergeClapScore,
  parseClapBatchOutput,
  shouldScoreRound
}
