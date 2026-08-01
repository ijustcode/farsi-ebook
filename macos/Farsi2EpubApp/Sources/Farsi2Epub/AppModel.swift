import AppKit
import Foundation

struct BookWorkspace: Identifiable, Hashable {
    let slug: String
    let root: URL
    var title: String
    var author: String
    var sourceType: String
    var pageCount: Int
    var pageRange: String
    var transcribedCount: Int
    var flaggedCount: Int
    var pendingQCCount: Int
    var epubURL: URL?

    var id: String { slug }
    var displayTitle: String { title.isEmpty ? slug : title }

    var progress: Double {
        let expected = expectedPageCount
        return expected == 0 ? 0 : min(1, Double(transcribedCount) / Double(expected))
    }

    var expectedPageCount: Int {
        guard !pageRange.isEmpty else { return pageCount }
        return Self.count(pageSpec: pageRange, total: pageCount)
    }

    private static func count(pageSpec: String, total: Int) -> Int {
        var pages = Set<Int>()
        for raw in pageSpec.split(separator: ",") {
            let part = raw.trimmingCharacters(in: .whitespaces)
            if let dash = part.firstIndex(of: "-") {
                let left = String(part[..<dash]).trimmingCharacters(in: .whitespaces)
                let right = String(part[part.index(after: dash)...]).trimmingCharacters(in: .whitespaces)
                let start = max(1, Int(left) ?? 1)
                let end = min(total, Int(right) ?? total)
                if start <= end { pages.formUnion(start...end) }
            } else if let page = Int(part), (1...total).contains(page) {
                pages.insert(page)
            }
        }
        return pages.count
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var repositoryURL: URL
    @Published var books: [BookWorkspace] = []
    @Published var selectedSlug: String?
    @Published var scanError: String?

    private let rootDefaultsKey = "Farsi2Epub.repositoryPath"

    init() {
        let defaults = UserDefaults.standard
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let bundledCandidate = Bundle.main.bundleURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        if let saved = defaults.string(forKey: rootDefaultsKey) {
            repositoryURL = URL(fileURLWithPath: saved)
        } else if FileManager.default.fileExists(atPath: cwd.appending(path: "farsi2epub/cli.py").path) {
            repositoryURL = cwd
        } else if FileManager.default.fileExists(atPath: bundledCandidate.appending(path: "farsi2epub/cli.py").path) {
            repositoryURL = bundledCandidate
        } else {
            repositoryURL = cwd
        }
        refresh()
    }

    var executableURL: URL { repositoryURL.appending(path: "venv/bin/farsi2epub") }
    var isRepositoryValid: Bool {
        FileManager.default.isExecutableFile(atPath: executableURL.path)
    }
    var selectedBook: BookWorkspace? { books.first { $0.slug == selectedSlug } }

    func setRepository(_ url: URL) {
        repositoryURL = url
        UserDefaults.standard.set(url.path, forKey: rootDefaultsKey)
        refresh()
    }

    func refresh(select slug: String? = nil) {
        let fm = FileManager.default
        guard isRepositoryValid else {
            books = []
            scanError = "Choose the farsi-ebook folder containing venv/bin/farsi2epub."
            return
        }
        scanError = nil
        let booksRoot = repositoryURL.appending(path: "books")
        let dirs = (try? fm.contentsOfDirectory(
            at: booksRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        books = dirs.compactMap(loadWorkspace).sorted {
            $0.displayTitle.localizedCaseInsensitiveCompare($1.displayTitle) == .orderedAscending
        }
        if let slug, books.contains(where: { $0.slug == slug }) {
            selectedSlug = slug
        } else if let selectedSlug, books.contains(where: { $0.slug == selectedSlug }) {
            self.selectedSlug = selectedSlug
        } else {
            selectedSlug = books.first?.slug
        }
    }

    private func loadWorkspace(_ root: URL) -> BookWorkspace? {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else { return nil }
        let metaURL = root.appending(path: "book.yaml")
        guard let yaml = try? String(contentsOf: metaURL, encoding: .utf8) else { return nil }
        let meta = parseSimpleYAML(yaml)
        let slug = meta["slug"] ?? root.lastPathComponent
        let textDir = root.appending(path: "text")
        let files = (try? FileManager.default.contentsOfDirectory(at: textDir, includingPropertiesForKeys: nil)) ?? []
        let mdStems = Set(files.filter { $0.pathExtension == "md" && Int($0.deletingPathExtension().lastPathComponent) != nil }.map { $0.deletingPathExtension().lastPathComponent })
        let jsonFiles = files.filter { $0.pathExtension == "json" && Int($0.deletingPathExtension().lastPathComponent) != nil }
        let jsonStems = Set(jsonFiles.map { $0.deletingPathExtension().lastPathComponent })
        var flagged = 0
        var pending = 0
        for file in jsonFiles {
            guard let data = try? Data(contentsOf: file),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if object["needs_review"] as? Bool == true || !((object["issues"] as? [Any]) ?? []).isEmpty { flagged += 1 }
            if let qc = object["qc"] as? [String: Any], (qc["status"] as? String) == "pending" { pending += 1 }
        }
        let epubs = ((try? FileManager.default.contentsOfDirectory(at: root.appending(path: "out"), includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension.lowercased() == "epub" }
        return BookWorkspace(
            slug: slug,
            root: root,
            title: clean(meta["title_fa"]),
            author: clean(meta["author_fa"]),
            sourceType: clean(meta["source_type"]),
            pageCount: Int(meta["page_count"] ?? "") ?? 0,
            pageRange: clean(meta["page_range"]),
            transcribedCount: mdStems.intersection(jsonStems).count,
            flaggedCount: flagged,
            pendingQCCount: pending,
            epubURL: epubs.sorted { $0.lastPathComponent < $1.lastPathComponent }.last
        )
    }

    private func parseSimpleYAML(_ source: String) -> [String: String] {
        var result: [String: String] = [:]
        for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
            guard !line.hasPrefix(" "), let colon = line.firstIndex(of: ":") else { continue }
            let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            result[key] = value.trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
        }
        return result
    }

    private func clean(_ value: String?) -> String {
        guard let value, value != "null", value != "~" else { return "" }
        return value
    }
}

struct CommandStep: Identifiable {
    let id = UUID()
    let label: String
    let arguments: [String]
}

@MainActor
final class CommandRunner: ObservableObject {
    @Published var isRunning = false
    @Published var currentLabel = ""
    @Published var log = ""
    @Published var completedSteps = 0
    @Published var totalSteps = 0
    @Published var lastExitCode: Int32?

    private var process: Process?
    private var task: Task<Void, Never>?
    var onFinish: (() -> Void)?

    func start(steps: [CommandStep], model: AppModel) {
        guard !isRunning, !steps.isEmpty else { return }
        isRunning = true
        log = ""
        completedSteps = 0
        totalSteps = steps.count
        lastExitCode = nil
        task = Task {
            for step in steps {
                if Task.isCancelled { break }
                currentLabel = step.label
                append("\n▸ \(step.label)\n$ farsi2epub \(shellDisplay(step.arguments))\n\n")
                let code = await run(step: step, model: model)
                lastExitCode = code
                if code != 0 {
                    append("\nCommand stopped with exit code \(code).\n")
                    break
                }
                completedSteps += 1
                model.refresh(select: model.selectedSlug)
            }
            currentLabel = ""
            isRunning = false
            process = nil
            onFinish?()
        }
    }

    func cancel() {
        task?.cancel()
        process?.interrupt()
        append("\nStopping…\n")
    }

    private func run(step: CommandStep, model: AppModel) async -> Int32 {
        await withCheckedContinuation { continuation in
            let process = Process()
            self.process = process
            process.executableURL = model.executableURL
            process.arguments = step.arguments
            process.currentDirectoryURL = model.repositoryURL
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONUNBUFFERED"] = "1"
            process.environment = environment

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let data = handle.availableData
                guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
                Task { @MainActor in self?.append(text) }
            }
            process.terminationHandler = { [weak self] process in
                pipe.fileHandleForReading.readabilityHandler = nil
                let remaining = pipe.fileHandleForReading.readDataToEndOfFile()
                Task { @MainActor in
                    if let text = String(data: remaining, encoding: .utf8), !text.isEmpty { self?.append(text) }
                    continuation.resume(returning: process.terminationStatus)
                }
            }
            do {
                try process.run()
            } catch {
                append("Could not launch the CLI: \(error.localizedDescription)\n")
                continuation.resume(returning: -1)
            }
        }
    }

    private func append(_ text: String) { log.append(text) }

    private func shellDisplay(_ args: [String]) -> String {
        args.map { value in
            value.rangeOfCharacter(from: .whitespacesAndNewlines) == nil ? value : "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
        }.joined(separator: " ")
    }
}

enum PipelineCommands {
    static func transcribe(slug: String, pages: String, force: Bool, maxCost: String, concurrency: Int, model: String, resolution: String) -> CommandStep {
        var args = ["transcribe", slug]
        if !pages.isEmpty { args += ["--pages", pages] }
        if force { args.append("--force") }
        if !maxCost.isEmpty { args += ["--max-cost", maxCost] }
        args += ["--concurrency", String(concurrency), "--res", resolution, "--qc", "skip"]
        if !model.isEmpty { args += ["--model", model] }
        return CommandStep(label: "Transcribing \(slug)", arguments: args)
    }

    static func qc(slug: String, pages: String, mode: String = "auto", all: Bool, yes: Bool, force: Bool) -> CommandStep {
        var args = ["qc", slug, "--mode", mode]
        if all { args.append("--all") }
        if yes { args.append("--yes") }
        if force { args.append("--force") }
        if !pages.isEmpty { args += ["--pages", pages] }
        return CommandStep(label: mode == "auto" ? "Running quality control" : "Opening manual QC", arguments: args)
    }

    static func review(slug: String, all: Bool, refine: Bool, model: String, background: Bool = false) -> CommandStep {
        var args = ["review", slug]
        if all { args.append("--all") }
        if background { args.append("--background") }
        if !refine { args.append("--no-bbox-refine") }
        if !model.isEmpty { args += ["--bbox-refine-model", model] }
        return CommandStep(label: "Opening manual review", arguments: args)
    }
}
