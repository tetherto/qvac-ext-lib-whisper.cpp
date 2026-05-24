import Foundation

struct QwenTranscription {
    let text: String
    let encodeMs: Double
    let decodeMs: Double
    let totalMs: Double
    let audioMs: Double
    let tokens: Int

    var realTimeFactor: Double {
        guard audioMs > 0 else { return 0 }
        return audioMs / totalMs
    }
}

enum QwenError: Error, LocalizedError {
    case engineCreate(String)
    case transcribe(String)

    var errorDescription: String? {
        switch self {
        case .engineCreate(let m): return "engine create: \(m)"
        case .transcribe(let m):   return "transcribe: \(m)"
        }
    }
}

final class QwenEngine {
    private var handle: OpaquePointer?

    init(modelDir: String, nThreads: Int32 = 0, verbose: Int32 = 1) throws {
        var opts = qwen_c_options_default()
        opts.n_threads = nThreads
        opts.verbose   = verbose

        var errorPtr: UnsafeMutablePointer<CChar>?
        let raw = modelDir.withCString { cstr in
            qwen_c_engine_create(cstr, opts, &errorPtr)
        }
        guard let raw = raw else {
            let msg = errorPtr.map { String(cString: $0) } ?? "unknown error"
            if let e = errorPtr { qwen_c_string_free(e) }
            throw QwenError.engineCreate(msg)
        }
        self.handle = raw
    }

    deinit {
        if let h = handle { qwen_c_engine_destroy(h) }
    }

    func transcribe(wavPath: String) throws -> QwenTranscription {
        guard let h = handle else { throw QwenError.transcribe("engine destroyed") }
        var result = qwen_c_result(text: nil, encode_ms: 0, decode_ms: 0,
                                   total_ms: 0, audio_ms: 0, text_tokens: 0)
        var errorPtr: UnsafeMutablePointer<CChar>?
        let rc = wavPath.withCString { wav in
            qwen_c_engine_transcribe(h, wav, &result, &errorPtr)
        }
        if rc != 0 {
            let msg = errorPtr.map { String(cString: $0) } ?? "unknown error"
            if let e = errorPtr { qwen_c_string_free(e) }
            throw QwenError.transcribe(msg)
        }
        let text = result.text.map { String(cString: $0) } ?? ""
        let out  = QwenTranscription(text:      text,
                                     encodeMs:  result.encode_ms,
                                     decodeMs:  result.decode_ms,
                                     totalMs:   result.total_ms,
                                     audioMs:   result.audio_ms,
                                     tokens:    Int(result.text_tokens))
        qwen_c_result_free(&result)
        return out
    }
}
