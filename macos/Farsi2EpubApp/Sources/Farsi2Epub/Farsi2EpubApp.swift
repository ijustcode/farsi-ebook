import SwiftUI

@main
struct Farsi2EpubApp: App {
    @StateObject private var model = AppModel()
    @StateObject private var runner = CommandRunner()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .environmentObject(runner)
                .frame(minWidth: 1050, minHeight: 700)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1220, height: 780)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Refresh Workspaces") { model.refresh() }
                    .keyboardShortcut("r", modifiers: [.command])
            }
        }
    }
}
