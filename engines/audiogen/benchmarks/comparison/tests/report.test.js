'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const { renderMarkdownReport } = require('../lib/report')

test('renderMarkdownReport includes medians not only fastest runs', () => {
  const markdown = renderMarkdownReport({
    meta: {
      generatedAt: '2026-08-19T00:00:00.000Z',
      platform: 'darwin-arm64',
      backend: 'cpu',
      comparisonClass: 'implementation-level',
      threads: 4,
      warmups: 1,
      runs: 3,
      order: 'alternate',
      cooldownMs: 0
    },
    overall: {
      qvac: {
        successCount: 3,
        failureCount: 0,
        generationMs: { median: 1000 },
        e2eMs: { median: 1100 },
        rtf: { median: 0.2 },
        peakRssBytes: { median: 1048576 },
        uniqueWavHashes: 3,
        silentCount: 0
      }
    },
    byPrompt: {}
  })
  assert.match(markdown, /implementation-level/)
  assert.match(markdown, /1000.0/)
  assert.match(markdown, /Median gen ms/)
  assert.match(markdown, /Median CLAP/)
})
