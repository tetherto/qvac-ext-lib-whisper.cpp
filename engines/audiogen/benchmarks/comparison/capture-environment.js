#!/usr/bin/env node
'use strict'

const os = require('os')
const { spawnSync } = require('child_process')
const { writeJson } = require('./lib/process')
const { loadHarnessConfig } = require('./lib/config')
const path = require('path')

function runText (command, args) {
  const result = spawnSync(command, args, { encoding: 'utf8' })
  if (result.status !== 0) return (result.stderr || result.stdout || '').trim()
  return (result.stdout || '').trim()
}

function main () {
  const config = loadHarnessConfig()
  const info = {
    generatedAt: new Date().toISOString(),
    hostname: os.hostname(),
    platform: process.platform,
    arch: process.arch,
    release: os.release(),
    cpus: os.cpus().map(cpu => cpu.model),
    cpuCount: os.cpus().length,
    totalMemBytes: os.totalmem(),
    node: process.version,
    cmake: runText('cmake', ['--version']),
    clang: runText('clang', ['--version']),
    sysctl: process.platform === 'darwin' ? {
      brand: runText('sysctl', ['-n', 'machdep.cpu.brand_string']),
      memsize: runText('sysctl', ['-n', 'hw.memsize']),
      machine: runText('sysctl', ['-n', 'hw.model'])
    } : null,
    swVers: process.platform === 'darwin' ? runText('sw_vers', []) : null,
    qvacHead: runText('git', ['-C', config.repositoryDir, 'rev-parse', 'HEAD']),
    acestepHead: fsExists(config.acestepDir)
      ? runText('git', ['-C', config.acestepDir, 'rev-parse', 'HEAD'])
      : null
  }
  const outPath = path.join(config.outDir, 'environment.json')
  writeJson(outPath, info)
  console.log(`Wrote ${outPath}`)
}

function fsExists (filePath) {
  return require('fs').existsSync(filePath)
}

try {
  main()
} catch (error) {
  console.error(`error: ${error.message}`)
  process.exitCode = 1
}
