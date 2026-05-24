import SwiftUI

private struct AudioSample: Identifiable, Hashable {
    let id:    String
    let label: String
    let flag:  String
    let basename: String

    var wavURL: URL? {
        Bundle.main.url(forResource: basename, withExtension: "wav", subdirectory: "samples")
        ?? Bundle.main.url(forResource: basename, withExtension: "wav")
    }
    var txtURL: URL? {
        Bundle.main.url(forResource: basename, withExtension: "txt", subdirectory: "samples")
        ?? Bundle.main.url(forResource: basename, withExtension: "txt")
    }

    var groundTruth: String {
        guard let u = txtURL, let s = try? String(contentsOf: u, encoding: .utf8) else {
            return ""
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private let SAMPLES: [AudioSample] = [
    .init(id: "en", label: "English (JFK)",       flag: "US", basename: "jfk"),
    .init(id: "es", label: "Spanish (LatAm)",     flag: "MX", basename: "es"),
    .init(id: "it", label: "Italian",             flag: "IT", basename: "it"),
    .init(id: "ru", label: "Russian",             flag: "RU", basename: "ru"),
    .init(id: "ar", label: "Arabic (Egyptian)",   flag: "EG", basename: "ar"),
    .init(id: "zh", label: "Chinese (Mandarin)",  flag: "CN", basename: "zh"),
    .init(id: "ja", label: "Japanese",            flag: "JP", basename: "ja"),
    .init(id: "ko", label: "Korean",              flag: "KR", basename: "ko"),
    .init(id: "hi", label: "Hindi",               flag: "IN", basename: "hi"),
]

private enum DemoState {
    case idle
    case loading
    case transcribing
    case done(QwenTranscription)
    case failed(String)
}

struct ContentView: View {
    @State private var selected: AudioSample = SAMPLES[0]
    @State private var state: DemoState = .idle
    @State private var engine: QwenEngine?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    Divider()
                    languagePicker
                    groundTruthCard
                    transcribeButton
                    Group {
                        switch state {
                        case .idle:          EmptyView()
                        case .loading:       loadingView("Loading model (mmap 1.87 GB safetensors)...")
                        case .transcribing:  loadingView("Transcribing on Apple Silicon (Accelerate BLAS)...")
                        case .failed(let m): errorView(m)
                        case .done(let r):   resultView(r)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Qwen3-ASR · 0.6B")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("On-device multilingual ASR demo")
                .font(.headline)
            Text("Backed by the vendored antirez/qwen-asr C kernels. Model: Qwen3-ASR-0.6B (bf16 safetensors, ~1.87 GB).")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var languagePicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Language sample").font(.subheadline).foregroundStyle(.secondary)
            Menu {
                ForEach(SAMPLES) { s in
                    Button {
                        selected = s
                        if case .done = state { state = .idle }
                        if case .failed = state { state = .idle }
                    } label: {
                        Label("\(s.flag)  \(s.label)", systemImage: s.id == selected.id ? "checkmark" : "")
                    }
                }
            } label: {
                HStack {
                    Text("\(selected.flag)  \(selected.label)")
                        .font(.body)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.footnote)
                }
                .padding(.vertical, 10)
                .padding(.horizontal, 12)
                .background(Color.gray.opacity(0.12))
                .cornerRadius(10)
            }
        }
    }

    private var groundTruthCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Ground truth (FLEURS)").font(.caption).foregroundStyle(.secondary)
            Text(selected.groundTruth.isEmpty ? "— no reference available —" : selected.groundTruth)
                .font(.system(.footnote, design: .serif))
                .multilineTextAlignment(.leading)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.blue.opacity(0.08))
                .cornerRadius(8)
        }
    }

    @ViewBuilder
    private var transcribeButton: some View {
        switch state {
        case .loading, .transcribing:
            ProgressView().frame(maxWidth: .infinity).padding(.vertical, 10)
        default:
            Button(action: runTranscription) {
                Label("Transcribe", systemImage: "waveform.badge.magnifyingglass")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
            }
            .buttonStyle(.borderedProminent)
        }
    }

    private func loadingView(_ msg: String) -> some View {
        HStack(spacing: 12) {
            ProgressView()
            Text(msg).font(.footnote)
        }
        .padding(.vertical, 4)
    }

    private func errorView(_ msg: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Error").font(.headline).foregroundStyle(.red)
            Text(msg).font(.system(.footnote, design: .monospaced))
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.red.opacity(0.08))
                .cornerRadius(8)
        }
    }

    private func resultView(_ r: QwenTranscription) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Transcript").font(.headline)
            Text(r.text)
                .font(.system(.body, design: .serif))
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.green.opacity(0.10))
                .cornerRadius(8)

            Text("Performance").font(.headline)
            metricsTable(r)
        }
    }

    private func metricsTable(_ r: QwenTranscription) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            metricRow("Tokens",  String(r.tokens))
            metricRow("Audio",   String(format: "%.0f ms", r.audioMs))
            metricRow("Encode",  String(format: "%.0f ms", r.encodeMs))
            metricRow("Decode",  String(format: "%.0f ms", r.decodeMs))
            metricRow("Total",   String(format: "%.0f ms", r.totalMs))
            HStack {
                Text("RTFx").foregroundStyle(.secondary).frame(width: 80, alignment: .leading)
                Text(String(format: "%.2fx realtime", r.realTimeFactor))
                    .bold()
                    .foregroundStyle(.green)
            }
        }
        .font(.system(.footnote, design: .monospaced))
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .cornerRadius(8)
    }

    private func metricRow(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).foregroundStyle(.secondary).frame(width: 80, alignment: .leading)
            Text(v)
        }
    }

    private func runTranscription() {
        guard let wavURL = selected.wavURL else {
            state = .failed("sample wav not found in bundle: \(selected.basename).wav")
            return
        }
        let wavPath = wavURL.path
        state = engine == nil ? .loading : .transcribing
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                if engine == nil {
                    let modelDir = ContentView.defaultModelDir()
                    let e = try QwenEngine(modelDir: modelDir, nThreads: 0, verbose: 0)
                    DispatchQueue.main.async {
                        self.engine = e
                        self.state  = .transcribing
                    }
                }
                let useEngine: QwenEngine = engine ?? {
                    let e = try! QwenEngine(modelDir: ContentView.defaultModelDir(),
                                            nThreads: 0, verbose: 0)
                    DispatchQueue.main.async { self.engine = e }
                    return e
                }()
                let r = try useEngine.transcribe(wavPath: wavPath)
                DispatchQueue.main.async { self.state = .done(r) }
            } catch {
                DispatchQueue.main.async { self.state = .failed(error.localizedDescription) }
            }
        }
    }

    private static func defaultModelDir() -> String {
        if let res = Bundle.main.resourceURL {
            let candidate = res.appendingPathComponent("0.6b").path
            if FileManager.default.fileExists(atPath: candidate) { return candidate }
        }
        return "0.6b"
    }
}

#Preview {
    ContentView()
}
