#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
package_dir="$repo_dir/macos/Farsi2EpubApp"
bundle_dir="$repo_dir/dist/Farsi2Epub.app"

swift build -c release --package-path "$package_dir"
binary_path="$(swift build -c release --package-path "$package_dir" --show-bin-path)/Farsi2Epub"

mkdir -p "$bundle_dir/Contents/MacOS"
mkdir -p "$bundle_dir/Contents/Resources"
cp "$binary_path" "$bundle_dir/Contents/MacOS/Farsi2Epub"

cat > "$bundle_dir/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>Farsi2Epub</string>
    <key>CFBundleIdentifier</key><string>dev.farsi2epub.app</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>Farsi2Epub</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

codesign --force --sign - "$bundle_dir"
echo "$bundle_dir"
