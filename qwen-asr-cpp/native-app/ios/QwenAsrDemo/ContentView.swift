import AVFoundation
import Darwin.Mach
import SwiftUI
import UniformTypeIdentifiers

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

private enum BackendState {
    case idle
    case loading
    case transcribing
    case done(QwenTranscription)
    case failed(String)
}

final class SamplePlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var isPlaying = false
    private var player: AVAudioPlayer?

    func toggle(url: URL?) {
        guard let url = url else { return }
        if isPlaying {
            stop()
            return
        }
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            let p = try AVAudioPlayer(contentsOf: url)
            p.delegate = self
            p.prepareToPlay()
            p.play()
            player = p
            isPlaying = true
        } catch {
            isPlaying = false
        }
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully _: Bool) {
        DispatchQueue.main.async { [weak self] in self?.isPlaying = false }
    }
}

struct ContentView: View {
    @State private var selected: AudioSample = SAMPLES[0]
    @State private var perBackend: [QwenBackend: BackendState] = [
        .safetensors: .idle,
        .gguf:        .idle
    ]
    @State private var engines: [QwenBackend: QwenEngine] = [:]
    @State private var isRunning: Bool = false
    @State private var currentlyRunning: QwenBackend?
    @State private var hasRunOnce: Bool = false
    @State private var pickFolderActive: Bool = false
    @State private var pickFileActive: Bool = false
    @State private var debugLog: [String] = []
    @StateObject private var player = SamplePlayer()
    @StateObject private var modelStore = ModelStore()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    Divider()
                    modelFilesSection
                    Divider()
                    languagePicker
                    groundTruthCard
                    transcribeButton
                    if hasRunOnce {
                        resultsSection
                    }
                    if !debugLog.isEmpty {
                        debugLogSection
                    }
                }
                .padding()
            }
            .navigationTitle("Qwen3-ASR · 0.6B")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $pickFolderActive) {
                NativeDocumentPicker(types: [.folder]) { urls in
                    pickFolderActive = false
                    handlePickedUrls(urls, role: .safetensors)
                } onCancel: {
                    pickFolderActive = false
                    log("pick(safetensors) cancelled")
                }
            }
            .sheet(isPresented: $pickFileActive) {
                NativeDocumentPicker(types: ggufContentTypes()) { urls in
                    pickFileActive = false
                    handlePickedUrls(urls, role: .gguf)
                } onCancel: {
                    pickFileActive = false
                    log("pick(gguf) cancelled")
                }
            }
        }
    }

    private func ggufContentTypes() -> [UTType] {
        if let t = UTType(filenameExtension: "gguf") { return [t, .data] }
        return [.data]
    }

    private func handlePickedUrls(_ urls: [URL], role: ModelRole) {
        log("pick(\(role.rawValue)) -> \(urls.count) url(s)")
        guard let url = urls.first else {
            log("pick(\(role.rawValue)) -> empty selection")
            return
        }
        log("pick(\(role.rawValue)) url=\(url.lastPathComponent)")
        Task { await applyPickedUrl(url, role: role) }
    }

    @MainActor
    private func applyPickedUrl(_ url: URL, role: ModelRole) async {
        let downloaded = await ensureLocallyAvailable(url, role: role)
        guard downloaded else { return }
        modelStore.setPath(url, for: role)
        if let err = modelStore.lastError[role] {
            log("setPath(\(role.rawValue)) ERROR: \(err)")
        } else if let p = modelStore.path(for: role) {
            log("setPath(\(role.rawValue)) OK path=\(URL(fileURLWithPath: p).lastPathComponent)")
        }
        resetEngine(for: role)
    }

    @MainActor
    private func ensureLocallyAvailable(_ url: URL, role: ModelRole) async -> Bool {
        let started = url.startAccessingSecurityScopedResource()
        let exists  = FileManager.default.fileExists(atPath: url.path)
        var sizeInfo = ""
        if exists, let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey]) {
            if values.isDirectory == true {
                sizeInfo = " (folder)"
            } else if let size = values.fileSize {
                sizeInfo = " (\(size / 1_000_000) MB)"
            }
        }
        if started { url.stopAccessingSecurityScopedResource() }
        if exists {
            log("\(role.rawValue) exists locally\(sizeInfo)")
            return true
        }
        log("\(role.rawValue) NOT downloaded yet; requesting iCloud download...")
        do {
            try FileManager.default.startDownloadingUbiquitousItem(at: url)
        } catch {
            log("startDownloadingUbiquitousItem failed: \(error.localizedDescription)")
            modelStore.lastError[role] = "iCloud download could not start: \(error.localizedDescription). In Files, long-press the item and tap 'Keep Downloaded'."
            return false
        }
        return await waitForDownload(url: url, role: role)
    }

    @MainActor
    private func waitForDownload(url: URL, role: ModelRole) async -> Bool {
        let deadline = Date().addingTimeInterval(120)
        while Date() < deadline {
            let started = url.startAccessingSecurityScopedResource()
            let exists  = FileManager.default.fileExists(atPath: url.path)
            if started { url.stopAccessingSecurityScopedResource() }
            if exists { return true }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        log("\(role.rawValue) download timed out after 120s")
        modelStore.lastError[role] = "iCloud download not finished (timeout). In Files, mark the item 'Keep Downloaded' and retry."
        return false
    }

    private func log(_ msg: String) {
        let ts = ISO8601DateFormatter().string(from: Date())
        let mem = Self.residentMemoryMB()
        let line = "[\(ts)] [mem \(Int(mem)) MB] \(msg)"
        debugLog.append(line)
        if debugLog.count > 80 { debugLog.removeFirst(debugLog.count - 80) }
        Self.appendToPersistedLog(line)
    }

    private static func residentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / 1024.0 / 1024.0
    }

    private static func appendToPersistedLog(_ line: String) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let url = docs.appendingPathComponent("qwen-debug.log")
        let data = (line + "\n").data(using: .utf8) ?? Data()
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: url, options: .atomic)
        }
    }

    private var debugLogSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Debug log").font(.caption).bold()
                Spacer()
                Button("Clear") { debugLog.removeAll() }
                    .font(.caption)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(debugLog.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption2, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .frame(maxHeight: 180)
            .padding(6)
            .background(Color.black.opacity(0.05))
            .cornerRadius(6)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("On-device multilingual ASR · backend A/B comparison")
                .font(.headline)
            Text("Safetensors (v0.1, vendored antirez/qwen-asr) and GGUF (v0.2, our GGML port) run side by side on the same audio.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var modelFilesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "folder.badge.gearshape")
                    .foregroundStyle(.secondary)
                Text("Step 1 · Model files on disk")
                    .font(.subheadline).bold()
                Spacer()
            }
            Text("Select once where each backend's model lives in Files / iCloud Drive. Paths are remembered between launches.")
                .font(.caption2)
                .foregroundStyle(.secondary)
            ForEach(ModelRole.allCases) { role in
                modelPickerRow(role)
            }
        }
        .padding(10)
        .background(Color.gray.opacity(0.06))
        .cornerRadius(8)
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "text.bubble")
                    .foregroundStyle(.secondary)
                Text("Step 3 · Results · side by side")
                    .font(.subheadline).bold()
                Spacer()
            }
            backendsGrid
        }
    }

    private func modelPickerRow(_ role: ModelRole) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: modelRoleIcon(role))
                    .foregroundStyle(modelRoleColor(role))
                Text(role.displayName).font(.footnote).bold()
                Spacer()
                if modelStore.isConfigured(role) {
                    Button("Change") { presentPicker(for: role) }
                        .font(.caption)
                    Button(role: .destructive) {
                        modelStore.clear(role)
                        resetEngine(for: role)
                        log("clear(\(role.rawValue))")
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless)
                } else {
                    Button("Select") { presentPicker(for: role) }
                        .font(.caption)
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                }
            }
            modelPathLabel(role)
        }
    }

    private func presentPicker(for role: ModelRole) {
        log("presentPicker(\(role.rawValue))")
        switch role {
        case .safetensors: pickFolderActive = true
        case .gguf:        pickFileActive   = true
        }
    }

    @ViewBuilder
    private func modelPathLabel(_ role: ModelRole) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if let p = modelStore.path(for: role) {
                Text(p)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
            } else {
                Text("not selected")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            if let err = modelStore.lastError[role] {
                Text(err)
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
    }

    private func modelRoleIcon(_ role: ModelRole) -> String {
        switch role {
        case .safetensors: return "shippingbox.fill"
        case .gguf:        return "bolt.fill"
        }
    }

    private func modelRoleColor(_ role: ModelRole) -> Color {
        switch role {
        case .safetensors: return .blue
        case .gguf:        return Color(red: 0.10, green: 0.45, blue: 0.20)
        }
    }

    private func backendRole(_ b: QwenBackend) -> ModelRole {
        switch b {
        case .safetensors: return .safetensors
        case .gguf:        return .gguf
        }
    }

    private func resetEngine(for role: ModelRole) {
        for b in QwenBackend.allCases where backendRole(b) == role {
            engines[b] = nil
            perBackend[b] = .idle
        }
    }

    private var languagePicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "waveform")
                    .foregroundStyle(.secondary)
                Text("Step 2 · Audio sample")
                    .font(.subheadline).bold()
                Spacer()
            }
            HStack(spacing: 8) {
                Menu {
                    ForEach(SAMPLES) { s in
                        Button {
                            if selected.id != s.id {
                                player.stop()
                            }
                            selected = s
                            resetResults()
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
                Button(action: { player.toggle(url: selected.wavURL) }) {
                    Image(systemName: player.isPlaying ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title)
                }
                .buttonStyle(.borderless)
                .accessibilityLabel(player.isPlaying ? "Stop preview" : "Play preview")
                .disabled(selected.wavURL == nil)
            }
            if let url = selected.wavURL {
                Text(url.lastPathComponent)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Text("\(selected.basename).wav (not bundled)")
                    .font(.caption2)
                    .foregroundStyle(.red)
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

    private var hasAnyModel: Bool {
        ModelRole.allCases.contains { modelStore.isConfigured($0) }
    }

    private var transcribeButton: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                ForEach(QwenBackend.allCases) { b in
                    backendRunButton(b)
                }
            }
            if !hasAnyModel {
                Text("Select at least one model above to enable transcribe.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
    }

    private func backendRunButton(_ b: QwenBackend) -> some View {
        let role = backendRole(b)
        let configured = modelStore.isConfigured(role)
        let isThisRunning = isRunning && currentlyRunning == b
        let label: String = {
            if isThisRunning { return "\(shortLabel(b))..." }
            return "Run \(shortLabel(b))"
        }()
        return Button {
            runOnly(b)
        } label: {
            Label(label, systemImage: isThisRunning ? "hourglass" : backendIcon(b))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)
        }
        .buttonStyle(.borderedProminent)
        .tint(backendColor(b))
        .disabled(isRunning || !configured)
    }

    private func shortLabel(_ b: QwenBackend) -> String {
        switch b {
        case .safetensors: return "Safetensors"
        case .gguf:        return "GGUF"
        }
    }

    private var backendsGrid: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 12) {
                ForEach(QwenBackend.allCases) { b in
                    backendCard(b)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                }
            }
            VStack(spacing: 12) {
                ForEach(QwenBackend.allCases) { b in
                    backendCard(b)
                }
            }
        }
    }

    private func backendCard(_ b: QwenBackend) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: backendIcon(b))
                    .foregroundStyle(backendColor(b))
                Text(b.displayName)
                    .font(.subheadline)
                    .bold()
                Spacer()
                statusBadge(b)
            }
            stateContent(b)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Color.gray.opacity(0.06))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(backendColor(b).opacity(0.25), lineWidth: 1)
        )
        .cornerRadius(10)
    }

    @ViewBuilder
    private func stateContent(_ b: QwenBackend) -> some View {
        switch perBackend[b] ?? .idle {
        case .idle:
            Text("Idle.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        case .loading:
            HStack(spacing: 8) {
                ProgressView()
                Text(loadingMessage(b)).font(.footnote)
            }
        case .transcribing:
            HStack(spacing: 8) {
                ProgressView()
                Text(transcribingMessage(b)).font(.footnote)
            }
        case .failed(let m):
            VStack(alignment: .leading, spacing: 4) {
                Text("Error").font(.caption).bold().foregroundStyle(.red)
                Text(m).font(.system(.caption, design: .monospaced))
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.red.opacity(0.08))
                    .cornerRadius(6)
            }
        case .done(let r):
            VStack(alignment: .leading, spacing: 8) {
                Text(r.text)
                    .font(.system(.footnote, design: .serif))
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.green.opacity(0.10))
                    .cornerRadius(6)
                metricsCompact(r)
            }
        }
    }

    @ViewBuilder
    private func statusBadge(_ b: QwenBackend) -> some View {
        switch perBackend[b] ?? .idle {
        case .idle:          Text("idle").font(.caption2).foregroundStyle(.secondary)
        case .loading:       Text("loading").font(.caption2).foregroundStyle(.orange)
        case .transcribing:  Text("running").font(.caption2).foregroundStyle(.orange)
        case .done:          Text("done").font(.caption2).foregroundStyle(.green)
        case .failed:        Text("failed").font(.caption2).foregroundStyle(.red)
        }
    }

    private func metricsCompact(_ r: QwenTranscription) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            metricRow("tokens", String(r.tokens))
            metricRow("audio",  String(format: "%.0f ms", r.audioMs))
            metricRow("encode", String(format: "%.0f ms", r.encodeMs))
            metricRow("decode", String(format: "%.0f ms", r.decodeMs))
            metricRow("total",  String(format: "%.0f ms", r.totalMs))
            HStack {
                Text("rtfx").foregroundStyle(.secondary).frame(width: 64, alignment: .leading)
                Text(String(format: "%.2fx", r.realTimeFactor))
                    .bold()
                    .foregroundStyle(.green)
            }
        }
        .font(.system(.caption2, design: .monospaced))
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.06))
        .cornerRadius(6)
    }

    private func metricRow(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).foregroundStyle(.secondary).frame(width: 64, alignment: .leading)
            Text(v)
        }
    }

    private func backendIcon(_ b: QwenBackend) -> String {
        switch b {
        case .safetensors: return "shippingbox.fill"
        case .gguf:        return "bolt.fill"
        }
    }

    private func backendColor(_ b: QwenBackend) -> Color {
        switch b {
        case .safetensors: return .blue
        case .gguf:        return Color(red: 0.10, green: 0.45, blue: 0.20)
        }
    }

    private func loadingMessage(_ b: QwenBackend) -> String {
        switch b {
        case .safetensors: return "Loading 1.87 GB safetensors (mmap)..."
        case .gguf:        return "Loading GGUF (GGML runtime)..."
        }
    }

    private func transcribingMessage(_ b: QwenBackend) -> String {
        switch b {
        case .safetensors: return "Transcribing (Accelerate BLAS)..."
        case .gguf:        return "Transcribing (GGML)..."
        }
    }

    private func resetResults() {
        for b in QwenBackend.allCases {
            perBackend[b] = .idle
        }
    }

    private func runOnly(_ b: QwenBackend) {
        guard let wavURL = selected.wavURL else {
            perBackend[b] = .failed("sample wav not found in bundle: \(selected.basename).wav")
            return
        }
        let wavPath = wavURL.path
        isRunning = true
        currentlyRunning = b
        hasRunOnce = true
        log("runOnly \(b.displayName) begin")
        runOnBackend(b, wavPath: wavPath) {
            autoreleasepool {
                self.engines[b] = nil
            }
            self.log("\(b.displayName) engine released (engines.count=\(self.engines.count))")
            self.isRunning = false
            self.currentlyRunning = nil
        }
    }

    private func runOnBackend(_ b: QwenBackend,
                              wavPath: String,
                              completion: @escaping () -> Void) {
        let role = backendRole(b)
        guard let modelPath = modelStore.path(for: role) else {
            perBackend[b] = .failed("\(role.displayName) not selected")
            log("skip \(b.displayName): not selected")
            completion()
            return
        }
        perBackend[b] = .loading
        log("[\(b.displayName)] engine_create begin: \(URL(fileURLWithPath: modelPath).lastPathComponent)")
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let e = try QwenEngine(modelPath: modelPath,
                                       backend: b,
                                       nThreads: 0,
                                       verbose: 0)
                DispatchQueue.main.async {
                    self.engines[b]    = e
                    self.perBackend[b] = .transcribing
                    self.log("[\(b.displayName)] engine_create OK; transcribe begin")
                }
                let r = try e.transcribe(wavPath: wavPath)
                DispatchQueue.main.async {
                    self.perBackend[b] = .done(r)
                    self.log("[\(b.displayName)] transcribe OK (\(r.tokens) tokens, \(Int(r.totalMs)) ms)")
                    completion()
                }
            } catch {
                DispatchQueue.main.async {
                    self.perBackend[b] = .failed(error.localizedDescription)
                    self.log("[\(b.displayName)] FAILED: \(error.localizedDescription)")
                    completion()
                }
            }
        }
    }
}


struct NativeDocumentPicker: UIViewControllerRepresentable {
    let types:    [UTType]
    let onPick:   ([URL]) -> Void
    let onCancel: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick, onCancel: onCancel)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: types, asCopy: false)
        picker.allowsMultipleSelection = false
        picker.shouldShowFileExtensions = true
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ vc: UIDocumentPickerViewController, context: Context) {}

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick:   ([URL]) -> Void
        let onCancel: () -> Void

        init(onPick: @escaping ([URL]) -> Void, onCancel: @escaping () -> Void) {
            self.onPick   = onPick
            self.onCancel = onCancel
        }

        func documentPicker(_ controller: UIDocumentPickerViewController,
                            didPickDocumentsAt urls: [URL]) {
            onPick(urls)
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            onCancel()
        }
    }
}

#Preview {
    ContentView()
}
