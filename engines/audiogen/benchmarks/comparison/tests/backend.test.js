'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const {
  assertBackend,
  canonicalBackend,
  detectAcestepBackend,
  detectQvacBackend
} = require('../lib/backend')

test('canonicalBackend maps Metal device names', () => {
  assert.equal(canonicalBackend('MTL0'), 'metal')
  assert.equal(canonicalBackend('ggml-metal'), 'metal')
  assert.equal(canonicalBackend('CPU'), 'cpu')
})

test('detectQvacBackend reads engine logs', () => {
  assert.equal(
    detectQvacBackend('[acestep-engine] DiT/VAE on GPU backend: MTL0'),
    'MTL0'
  )
  assert.equal(
    detectQvacBackend('GPU requested but no GPU backend available; using CPU'),
    'cpu-fallback'
  )
})

test('detectAcestepBackend flags CPU fallback', () => {
  assert.equal(detectAcestepBackend('falling back to CPU'), 'cpu-fallback')
  assert.equal(detectAcestepBackend('using ggml-metal'), 'Metal')
})

test('assertBackend fails closed on GPU fallback', () => {
  assert.throws(
    () => assertBackend('cpu-fallback', 'metal'),
    /GPU-to-CPU fallback/
  )
  assert.doesNotThrow(() => assertBackend('MTL0', 'metal'))
})
