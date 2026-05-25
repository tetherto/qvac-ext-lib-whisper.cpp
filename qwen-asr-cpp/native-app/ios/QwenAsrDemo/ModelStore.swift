import Foundation

enum ModelRole: String, CaseIterable, Identifiable {
    case safetensors
    case gguf

    var id: String { rawValue }

    var bookmarkKey: String {
        switch self {
        case .safetensors: return "model.safetensors.bookmark.v1"
        case .gguf:        return "model.gguf.bookmark.v1"
        }
    }

    var displayName: String {
        switch self {
        case .safetensors: return "Safetensors folder"
        case .gguf:        return "GGUF file"
        }
    }
}

private struct ResolvedModel {
    let url:  URL
    let path: String
}

@MainActor
final class ModelStore: ObservableObject {
    @Published private(set) var paths: [ModelRole: String] = [:]
    @Published var lastError: [ModelRole: String] = [:]

    private var openedURLs: [ModelRole: URL] = [:]
    private let defaults = UserDefaults.standard

    init() {
        ModelRole.allCases.forEach(restoreBookmark(for:))
    }

    deinit {
        openedURLs.values.forEach { $0.stopAccessingSecurityScopedResource() }
    }

    func path(for role: ModelRole) -> String? { paths[role] }
    func isConfigured(_ role: ModelRole) -> Bool { paths[role] != nil }

    func setPath(_ url: URL, for role: ModelRole) {
        clear(role)
        do {
            try beginAccess(url: url, for: role)
            try persistBookmark(url: url, for: role)
            lastError[role] = nil
        } catch {
            if let opened = openedURLs[role] {
                opened.stopAccessingSecurityScopedResource()
                openedURLs[role] = nil
            }
            paths[role] = nil
            defaults.removeObject(forKey: role.bookmarkKey)
            lastError[role] = error.localizedDescription
        }
    }

    func clear(_ role: ModelRole) {
        if let u = openedURLs[role] {
            u.stopAccessingSecurityScopedResource()
        }
        openedURLs[role] = nil
        paths[role] = nil
        defaults.removeObject(forKey: role.bookmarkKey)
    }

    private func persistBookmark(url: URL, for role: ModelRole) throws {
        let data = try url.bookmarkData(
            options: .minimalBookmark,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        defaults.set(data, forKey: role.bookmarkKey)
    }

    private func beginAccess(url: URL, for role: ModelRole) throws {
        let ok = url.startAccessingSecurityScopedResource()
        guard ok else {
            throw ModelStoreError.accessDenied(url: url)
        }
        openedURLs[role] = url
        paths[role] = url.path
    }

    private func restoreBookmark(for role: ModelRole) {
        guard let data = defaults.data(forKey: role.bookmarkKey) else { return }
        do {
            var stale = false
            let url = try URL(
                resolvingBookmarkData: data,
                options: [],
                relativeTo: nil,
                bookmarkDataIsStale: &stale
            )
            try beginAccess(url: url, for: role)
            if stale {
                try persistBookmark(url: url, for: role)
            }
        } catch {
            lastError[role] = "restore: \(error.localizedDescription)"
            defaults.removeObject(forKey: role.bookmarkKey)
        }
    }
}

enum ModelStoreError: LocalizedError {
    case accessDenied(url: URL)

    var errorDescription: String? {
        switch self {
        case .accessDenied(let url):
            return "no permission to access \(url.lastPathComponent)"
        }
    }
}
