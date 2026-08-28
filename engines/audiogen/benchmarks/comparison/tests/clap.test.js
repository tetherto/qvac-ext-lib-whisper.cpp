'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const {
  buildClapText,
  mergeClapScore,
  parseClapBatchOutput,
  shouldScoreRound
} = require('../lib/clap')
const { aggregateRounds } = require('../lib/aggregate')

test('buildClapText uses caption or caption plus lyrics', () => {
  const prompt = { caption: 'folk guitar', lyrics: '[Verse]\nHello' }
  assert.equal(buildClapText(prompt, 'caption'), 'folk guitar')
  assert.equal(buildClapText(prompt, 'caption+lyrics'), 'folk guitar\n[Verse]\nHello')
  assert.throws(() => buildClapText(prompt, 'tokens'), /unknown CLAP text policy/)
})

test('parseClapBatchOutput requires ok scores JSON', () => {
  const parsed = parseClapBatchOutput('{"ok":true,"scores":[{"id":"a","ok":true,"score":0.4}]}')
  assert.equal(parsed.scores[0].score, 0.4)
  assert.throws(() => parseClapBatchOutput(''), /no JSON/)
  assert.throws(() => parseClapBatchOutput('{"ok":false,"error":"missing torch"}'), /missing torch/)
})

test('shouldScoreRound skips warmups and existing scores unless forced', () => {
  const timed = { kind: 'timed', wavPath: '/tmp/a.wav' }
  assert.equal(shouldScoreRound(timed, { timedOnly: true, force: false }), true)
  assert.equal(shouldScoreRound({ kind: 'warmup', wavPath: '/tmp/a.wav' }, { timedOnly: true, force: false }), false)
  assert.equal(shouldScoreRound({ ...timed, clap: { score: 0.2 } }, { timedOnly: true, force: false }), false)
  assert.equal(shouldScoreRound({ ...timed, clap: { score: 0.2 } }, { timedOnly: true, force: true }), true)
})

test('mergeClapScore keeps generation times untouched', () => {
  const merged = mergeClapScore(
    { engine: 'qvac', generationMs: 1000, e2eMs: 1100 },
    { ok: true, score: 0.31, elapsedMs: 50 },
    { model: 'laion/larger_clap_music_and_speech', revision: 'abc', samplingRate: 48000, device: 'cpu', textPolicy: 'caption' }
  )
  assert.equal(merged.generationMs, 1000)
  assert.equal(merged.e2eMs, 1100)
  assert.equal(merged.clap.score, 0.31)
  assert.equal(merged.clap.elapsedMs, 50)
})

test('aggregateRounds summarises CLAP separately from RTF', () => {
  const stats = aggregateRounds([
    { engine: 'qvac', ok: true, generationMs: 10, e2eMs: 12, rtf: 0.5, clap: { score: 0.2 }, audio: { durationSeconds: 8, silenceRatio: 0, clippingRatio: 0 } },
    { engine: 'qvac', ok: true, generationMs: 14, e2eMs: 16, rtf: 0.6, clap: { score: 0.4 }, audio: { durationSeconds: 8, silenceRatio: 0, clippingRatio: 0 } }
  ]).qvac
  assert.ok(Math.abs(stats.clap.median - 0.3) < 1e-12)
  assert.ok(Math.abs(stats.rtf.median - 0.55) < 1e-12)
})
