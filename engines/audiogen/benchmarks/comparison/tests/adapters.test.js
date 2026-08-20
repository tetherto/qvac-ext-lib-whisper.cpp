'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const { buildRequest } = require('../adapters/acestep')
const { buildQvacArgs } = require('../adapters/qvac')

const config = {
  modelsDir: '/models',
  models: {
    text: 'Qwen3-Embedding-0.6B-Q8_0.gguf',
    lm: 'acestep-5Hz-lm-0.6B-Q8_0.gguf',
    dit: 'acestep-v15-turbo-Q8_0.gguf',
    vae: 'vae-BF16.gguf'
  },
  threads: 4,
  backend: 'cpu',
  parameters: {
    inference_steps: 8,
    shift: 3,
    guidance_scale: 1,
    solver: 'euler',
    lm_temperature: 0.85,
    lm_cfg_scale: 2,
    lm_top_p: 0.9,
    lm_top_k: 0,
    use_cot_caption: false,
    dcw_enabled: true,
    dcw_mode: 'double',
    dcw_scaler: 0.05,
    dcw_high_scaler: 0.02,
    output_format: 'wav16',
    peak_clip: 0,
    task_type: 'text2music'
  }
}

const prompt = {
  caption: 'test caption',
  lyrics: '[Instrumental]',
  duration: 8,
  bpm: 120,
  keyscale: 'C major',
  timesignature: '4',
  vocal_language: 'en',
  seed: 42
}

test('acestep request maps lm_seed and wav16', () => {
  const request = buildRequest(config, prompt)
  assert.equal(request.seed, 42)
  assert.equal(request.lm_seed, 42)
  assert.equal(request.output_format, 'wav16')
  assert.equal(request.peak_clip, 0)
  assert.equal(request.dcw_mode, 'double')
  assert.equal(request.lm_mode, 'generate')
  assert.equal(request.lm_model, 'acestep-5Hz-lm-0.6B-Q8_0.gguf')
  assert.equal(request.synth_model, 'acestep-v15-turbo-Q8_0.gguf')
  assert.equal(request.vae, 'vae-BF16.gguf')
})

test('qvac args pass explicit metadata and do not enable GPU on cpu', () => {
  const args = buildQvacArgs(config, prompt, '/tmp/out.wav')
  assert.equal(args.includes('--gpu'), false)
  assert.equal(args[args.indexOf('--seed') + 1], '42')
  assert.equal(args[args.indexOf('--dur') + 1], '8')
  assert.equal(args[args.indexOf('--tsig') + 1], '4')
})
