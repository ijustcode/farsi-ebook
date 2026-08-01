import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var runner: CommandRunner
    @State private var showingNewBook = false
    @State private var showingRepositoryPicker = false

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            VStack(spacing: 0) {
                if let book = model.selectedBook {
                    BookDetailView(book: book)
                } else {
                    emptyState
                }
                if runner.isRunning || !runner.log.isEmpty { RunConsoleView() }
            }
        }
        .toolbar {
            ToolbarItemGroup {
                Button { model.refresh() } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                Button { showingNewBook = true } label: { Label("New Book", systemImage: "plus") }
                    .disabled(!model.isRepositoryValid || runner.isRunning)
            }
        }
        .sheet(isPresented: $showingNewBook) { NewBookView(isPresented: $showingNewBook) }
        .fileImporter(isPresented: $showingRepositoryPicker, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { model.setRepository(url) }
        }
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            List(model.books, selection: $model.selectedSlug) { book in
                VStack(alignment: .leading, spacing: 5) {
                    Text(book.displayTitle).font(.headline).lineLimit(1)
                    HStack {
                        Text("\(book.transcribedCount)/\(book.expectedPageCount) pages")
                        Spacer()
                        if book.flaggedCount > 0 {
                            Label("\(book.flaggedCount)", systemImage: "exclamationmark.circle.fill")
                                .foregroundStyle(.orange)
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    ProgressView(value: book.progress).controlSize(.mini)
                }
                .padding(.vertical, 4)
                .tag(book.slug)
            }
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                Text("PROJECT FOLDER").font(.caption2.bold()).foregroundStyle(.secondary)
                Text(model.repositoryURL.path).font(.caption).lineLimit(2).truncationMode(.middle)
                Button("Choose Folder…") { showingRepositoryPicker = true }.buttonStyle(.link)
            }
            .padding(12)
        }
        .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label(model.isRepositoryValid ? "No Books Yet" : "Project Folder Needed", systemImage: "books.vertical")
        } description: {
            Text(model.scanError ?? "Analyze a PDF to create your first book workspace.")
        } actions: {
            if model.isRepositoryValid {
                Button("Add a PDF") { showingNewBook = true }.buttonStyle(.borderedProminent)
            } else {
                Button("Choose Project Folder…") { showingRepositoryPicker = true }.buttonStyle(.borderedProminent)
            }
        }
    }
}

enum BookSection: String, CaseIterable, Identifiable {
    case overview = "Overview"
    case transcribe = "Transcribe"
    case qc = "Quality Control"
    case review = "Review"
    case build = "Build"
    var id: String { rawValue }
}

struct BookDetailView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var runner: CommandRunner
    let book: BookWorkspace
    @State private var section: BookSection = .overview

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            Picker("Section", selection: $section) {
                ForEach(BookSection.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding()

            ScrollView {
                Group {
                    switch section {
                    case .overview: OverviewView(book: book, section: $section)
                    case .transcribe: TranscribeView(book: book)
                    case .qc: QCView(book: book)
                    case .review: ReviewView(book: book)
                    case .build: BuildView(book: book)
                    }
                }
                .padding(.horizontal, 22)
                .padding(.bottom, 18)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            Image(systemName: "book.closed.fill")
                .font(.system(size: 28)).foregroundStyle(.indigo)
                .frame(width: 48, height: 48).background(.indigo.opacity(0.12), in: RoundedRectangle(cornerRadius: 11))
            VStack(alignment: .leading, spacing: 3) {
                Text(book.displayTitle).font(.title2.bold())
                HStack(spacing: 10) {
                    Text(book.slug).monospaced()
                    if !book.author.isEmpty { Text(book.author) }
                    if !book.sourceType.isEmpty { Text(book.sourceType.capitalized) }
                }.font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if book.epubURL != nil { Label("EPUB ready", systemImage: "checkmark.seal.fill").foregroundStyle(.green) }
        }
        .padding(18)
    }
}

struct OverviewView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var runner: CommandRunner
    let book: BookWorkspace
    @Binding var section: BookSection
    @State private var pages = ""
    @State private var maxCost = ""
    @State private var resolution = "hi"
    @State private var fullQC = true
    @State private var fullReview = true

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 12) {
                MetricCard(title: "Transcribed", value: "\(book.transcribedCount) / \(book.expectedPageCount)", icon: "text.page")
                MetricCard(title: "Needs attention", value: "\(book.flaggedCount)", icon: "exclamationmark.triangle")
                MetricCard(title: "QC pending", value: "\(book.pendingQCCount)", icon: "checklist")
                MetricCard(title: "Output", value: book.epubURL == nil ? "Not built" : "Ready", icon: "book.pages")
            }
            GroupBox("Guided workflow") {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Runs transcription, automated QC, then opens the manual review in your browser. Completed pages are skipped, so this is safe to resume.")
                        .foregroundStyle(.secondary)
                    HStack {
                        LabeledContent("Pages") { TextField(book.pageRange.isEmpty ? "All pages" : book.pageRange, text: $pages).frame(width: 150) }
                        LabeledContent("Max transcription cost ($)") { TextField("Optional", text: $maxCost).frame(width: 100) }
                        LabeledContent("Resolution") {
                            Picker("", selection: $resolution) { Text("Best quality").tag("hi"); Text("Economy").tag("std") }.frame(width: 130)
                        }
                    }
                    HStack {
                        Toggle("QC every transcribed page", isOn: $fullQC)
                        Toggle("Show every flagged page in review", isOn: $fullReview)
                        Spacer()
                        Button("Run transcription → QC → review") { runGuided() }
                            .buttonStyle(.borderedProminent).disabled(runner.isRunning)
                    }
                }.padding(8)
            }
            GroupBox("Individual stages") {
                HStack(spacing: 10) {
                    StageButton(title: "Transcribe", subtitle: "Images → Persian text", icon: "text.viewfinder") { section = .transcribe }
                    StageButton(title: "Quality Control", subtitle: "Verify and suggest fixes", icon: "checkmark.circle") { section = .qc }
                    StageButton(title: "Manual Review", subtitle: "Approve corrections", icon: "rectangle.and.pencil.and.ellipsis") { section = .review }
                    StageButton(title: "Build EPUB", subtitle: "Assemble the ebook", icon: "hammer") { section = .build }
                }.padding(8)
            }
        }
        .onAppear { pages = book.pageRange }
        .onChange(of: book.slug) { _, _ in pages = book.pageRange }
    }

    private func runGuided() {
        let scope = pages.trimmingCharacters(in: .whitespaces)
        var steps = [PipelineCommands.transcribe(slug: book.slug, pages: scope, force: false, maxCost: maxCost, concurrency: 4, model: "", resolution: resolution)]
        steps.append(PipelineCommands.qc(slug: book.slug, pages: scope, all: fullQC, yes: true, force: false))
        steps.append(PipelineCommands.review(slug: book.slug, all: fullReview, refine: true, model: ""))
        runner.start(steps: steps, model: model)
    }
}

struct TranscribeView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var runner: CommandRunner
    let book: BookWorkspace
    @State private var pages = ""
    @State private var maxCost = ""
    @State private var concurrency = 4
    @State private var resolution = "hi"
    @State private var modelName = ""
    @State private var force = false

    var body: some View {
        Form {
            Section("Scope") {
                TextField("Pages (for example 1-100 or 1,3-5)", text: $pages)
                Text("Leave blank to use the page range saved in book.yaml, or all pages if none is saved.").font(.caption).foregroundStyle(.secondary)
                Toggle("Re-transcribe pages that already have output", isOn: $force)
            }
            Section("Cost and quality") {
                TextField("Maximum transcription spend in US dollars (optional)", text: $maxCost)
                Picker("Input resolution", selection: $resolution) { Text("High — best character accuracy").tag("hi"); Text("Standard — ~30% cheaper").tag("std") }
                Stepper("Concurrent pages: \(concurrency)", value: $concurrency, in: 1...12)
                TextField("Model override (leave blank for Sonnet default)", text: $modelName)
                Text("The cost limit applies only to transcription and can overshoot by pages already in flight. QC and box refinement are billed separately.").font(.caption).foregroundStyle(.secondary)
            }
            HStack { Spacer(); Button("Start Transcription") { run() }.buttonStyle(.borderedProminent).disabled(runner.isRunning) }
        }
        .formStyle(.grouped)
        .onAppear { pages = book.pageRange }
    }

    private func run() {
        runner.start(steps: [PipelineCommands.transcribe(slug: book.slug, pages: pages, force: force, maxCost: maxCost, concurrency: concurrency, model: modelName, resolution: resolution)], model: model)
    }
}

struct QCView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var runner: CommandRunner
    let book: BookWorkspace
    @State private var pages = ""
    @State private var allPages = true
    @State private var force = false

    var body: some View {
        Form {
            Section("Automated verification") {
                TextField("Pages (blank means all transcribed pages)", text: $pages)
                Toggle("Verify every transcribed page", isOn: $allPages)
                Text(allPages ? "Full coverage: every page in the selected scope is checked." : "Risk-selected: forced, risky, and sampled clean pages are checked.").font(.caption).foregroundStyle(.secondary)
                Toggle("Replace still-pending QC suggestions", isOn: $force)
                Text("Starting from this screen confirms the QC spend. The CLI's estimate and actual page costs appear in the log.").font(.caption).foregroundStyle(.secondary)
            }
            HStack { Spacer(); Button("Run Automated QC") { run() }.buttonStyle(.borderedProminent).disabled(runner.isRunning) }
        }.formStyle(.grouped)
    }

    private func run() {
        runner.start(steps: [PipelineCommands.qc(slug: book.slug, pages: pages, all: allPages, yes: true, force: force)], model: model)
    }
}

struct ReviewView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var runner: CommandRunner
    let book: BookWorkspace
    @State private var allPages = true
    @State private var refine = true
    @State private var refineModel = ""

    var body: some View {
        Form {
            Section("Manual review") {
                Toggle("Show every flagged page", isOn: $allPages)
                Text(allPages ? "No review budget: every flagged page is surfaced." : "Budgeted review: only the worst fifth is surfaced; the remainder is marked skipped.").font(.caption).foregroundStyle(.secondary)
                Toggle("Refine scan-page boxes with the vision model", isOn: $refine)
                TextField("Box-refinement model override (optional)", text: $refineModel).disabled(!refine)
            }
            HStack {
                Button("Server Status") { simple(["review", book.slug, "--status"], "Checking review server") }
                Button("Stop Server") { simple(["review", book.slug, "--stop"], "Stopping review server") }
                Button("Reset Decisions") { simple(["review", book.slug, "--reset"], "Resetting review decisions") }
                Spacer()
                Button("Open Manual Review") { openReview() }.buttonStyle(.borderedProminent)
            }.disabled(runner.isRunning)
        }.formStyle(.grouped)
    }

    private func openReview() {
        runner.start(steps: [PipelineCommands.review(slug: book.slug, all: allPages, refine: refine, model: refineModel)], model: model)
    }
    private func simple(_ args: [String], _ label: String) {
        runner.start(steps: [CommandStep(label: label, arguments: args)], model: model)
    }
}

struct BuildView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var runner: CommandRunner
    let book: BookWorkspace

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            GroupBox("EPUB output") {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Build joins page text, applies conservative Persian normalization, creates RTL chapters, embeds Vazirmatn, and runs epubcheck when available.")
                    if let url = book.epubURL {
                        HStack {
                            Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
                            Text(url.lastPathComponent).font(.headline)
                            Spacer()
                            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                            Button("Open") { NSWorkspace.shared.open(url) }
                        }
                    } else { Text("No EPUB has been built yet.").foregroundStyle(.secondary) }
                }.padding(8)
            }
            HStack { Spacer(); Button("Build EPUB") { runner.start(steps: [CommandStep(label: "Building EPUB", arguments: ["build", book.slug])], model: model) }.buttonStyle(.borderedProminent).disabled(runner.isRunning || book.transcribedCount == 0) }
        }
    }
}

struct NewBookView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var runner: CommandRunner
    @Binding var isPresented: Bool
    @State private var pdfURL: URL?
    @State private var slug = ""
    @State private var pages = ""
    @State private var overwrite = false
    @State private var continueWorkflow = true
    @State private var maxCost = ""
    @State private var resolution = "hi"
    @State private var choosingPDF = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Add a Farsi book").font(.title2.bold())
            GroupBox {
                HStack {
                    Image(systemName: "doc.richtext").font(.title2).foregroundStyle(.indigo)
                    VStack(alignment: .leading) {
                        Text(pdfURL?.lastPathComponent ?? "Choose a PDF")
                        Text(pdfURL?.deletingLastPathComponent().path ?? "The source is copied into its own workspace.").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer(); Button("Choose…") { choosingPDF = true }
                }.padding(8)
            }
            Form {
                TextField("Workspace name", text: $slug)
                TextField("Pages (for example 1-100; blank means the whole book)", text: $pages)
                Toggle("Overwrite a workspace with the same name", isOn: $overwrite)
                Toggle("Continue through transcription, full QC, and manual review", isOn: $continueWorkflow)
                if continueWorkflow {
                    TextField("Maximum transcription spend in US dollars (optional)", text: $maxCost)
                    Picker("Resolution", selection: $resolution) { Text("High — best quality").tag("hi"); Text("Standard — economy").tag("std") }
                }
            }.formStyle(.grouped)
            HStack {
                Button("Cancel") { isPresented = false }
                Spacer()
                Button(continueWorkflow ? "Analyze and Start" : "Analyze") { start() }
                    .buttonStyle(.borderedProminent).disabled(pdfURL == nil || slug.isEmpty || runner.isRunning)
            }
        }
        .padding(24).frame(width: 600)
        .fileImporter(isPresented: $choosingPDF, allowedContentTypes: [.pdf]) { result in
            if case .success(let url) = result {
                pdfURL = url
                if slug.isEmpty { slug = slugify(url.deletingPathExtension().lastPathComponent) }
            }
        }
    }

    private func start() {
        guard let pdfURL else { return }
        var analyze = ["analyze", pdfURL.path, "--slug", slug]
        if !pages.isEmpty { analyze += ["--pages", pages] }
        if overwrite { analyze.append("--force") }
        var steps = [CommandStep(label: "Analyzing \(pdfURL.lastPathComponent)", arguments: analyze)]
        if continueWorkflow {
            steps.append(PipelineCommands.transcribe(slug: slug, pages: pages, force: false, maxCost: maxCost, concurrency: 4, model: "", resolution: resolution))
            steps.append(PipelineCommands.qc(slug: slug, pages: pages, all: true, yes: true, force: false))
            steps.append(PipelineCommands.review(slug: slug, all: true, refine: true, model: ""))
        }
        model.selectedSlug = slug
        runner.start(steps: steps, model: model)
        isPresented = false
    }

    private func slugify(_ value: String) -> String {
        value.lowercased().replacingOccurrences(of: "[^a-z0-9]+", with: "-", options: .regularExpression).trimmingCharacters(in: CharacterSet(charactersIn: "-"))
    }
}

struct RunConsoleView: View {
    @EnvironmentObject private var runner: CommandRunner
    @State private var expanded = true

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            HStack {
                if runner.isRunning { ProgressView().controlSize(.small) }
                Text(runner.isRunning ? runner.currentLabel : (runner.lastExitCode == 0 ? "Finished" : "Command output")).font(.headline)
                if runner.totalSteps > 1 { Text("\(runner.completedSteps)/\(runner.totalSteps)").foregroundStyle(.secondary) }
                Spacer()
                Button(expanded ? "Hide Log" : "Show Log") { expanded.toggle() }.buttonStyle(.plain)
                if runner.isRunning { Button("Stop") { runner.cancel() }.foregroundStyle(.red) }
            }.padding(.horizontal, 16).padding(.vertical, 9)
            if expanded {
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(runner.log.isEmpty ? "Waiting for output…" : runner.log)
                            .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading).padding(12)
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .background(Color(nsColor: .textBackgroundColor))
                    .onChange(of: runner.log) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                }.frame(height: 190)
            }
        }
    }
}

struct MetricCard: View {
    let title: String, value: String, icon: String
    var body: some View {
        HStack {
            Image(systemName: icon).font(.title2).foregroundStyle(.indigo).frame(width: 34)
            VStack(alignment: .leading) { Text(value).font(.title3.bold()); Text(title).font(.caption).foregroundStyle(.secondary) }
            Spacer()
        }.padding(12).frame(maxWidth: .infinity).background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }
}

struct StageButton: View {
    let title: String, subtitle: String, icon: String, action: () -> Void
    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                Image(systemName: icon).font(.title2).foregroundStyle(.indigo)
                Text(title).font(.headline)
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading).padding(12)
        }.buttonStyle(.plain).background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 10))
    }
}
