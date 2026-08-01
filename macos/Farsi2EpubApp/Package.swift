// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Farsi2EpubApp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Farsi2Epub", targets: ["Farsi2Epub"])
    ],
    targets: [
        .executableTarget(
            name: "Farsi2Epub",
            path: "Sources/Farsi2Epub"
        )
    ]
)
