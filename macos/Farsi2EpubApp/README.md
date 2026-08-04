# Farsi2Epub for macOS

A native SwiftUI front end for the existing `farsi2epub` CLI. The app does not
duplicate pipeline logic: it runs the editable installation in `venv/`, streams
its output, and refreshes the existing `books/` workspaces after every stage.

## Run while developing

From the repository root:

```bash
swift run --package-path macos/Farsi2EpubApp Farsi2Epub
```

The app initially uses the current directory as the project folder. If it was
launched elsewhere, choose the repository folder at the bottom of the sidebar.

## Coverage

- Analyze a PDF, select a page range, choose a slug, and optionally overwrite.
- Guided analyze → transcribe → full auto-QC → full manual-review workflow.
- Resume or force transcription, with page range, dollar guard, concurrency,
  resolution, and model override.
- Automated QC with page range, full/risk-selected coverage, one-click
  noninteractive execution, and pending-suggestion replacement controls.
- Manual review with full/budgeted selection and bbox-refinement controls
  (on/off, model override, and algorithm choice — context anchor by default,
  legacy available as the superseded control), plus server status, stop, and
  decision reset.
- Build and open the resulting EPUB.

The manual review remains the project’s purpose-built browser UI; the native app
starts and monitors its local server so the page/correction interface does not
need to be duplicated.
