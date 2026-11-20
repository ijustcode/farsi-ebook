"""Kindle edition: convert the built EPUB to KFX (Amazon's Enhanced Typesetting format).

Why (verified on the owner's Kindle Paperwhite 2015, firmware 5.16.2.1.1, 2026-11-20):
Send to Kindle delivers our EPUB to that device as AZW3, whose renderer does not
join Persian letters drawn in an embedded font, so with "Publisher Font" every
letter shows isolated. The same EPUB converted to KFX by Calibre's KFX Output
plugin (which drives Kindle Previewer) and copied over USB shows joined letters
in Vazirmatn with Publisher Font selected.

This step only reads the EPUB; it never changes it, so the iPad / Apple Books /
other EPUB readers path is untouched. The KFX is written atomically next to the
EPUB, and only after it passes `verify_kfx`; a stale KFX from an earlier build is
removed whenever a new one is not produced, so the two never drift apart.

Owner decision D29 (docs/bbox-old-fonts-plan.md §10): the KFX is built by default
(`--kfx require`); the EPUB stays the file for the iPad and other EPUB readers.

Delivery: copy the .kfx into the Kindle's `documents/` folder over USB.
Send to Kindle does not accept KFX.

Tools (none are Python dependencies): calibre's `ebook-convert` with the
third-party "KFX Output" plugin (jhowell), and Kindle Previewer 4.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("auto", "require", "skip")
DEFAULT_MODE = "require"          # owner decision D29: the Kindle edition is part of every build
CONVERT_TIMEOUT_S = 900

_MAC_CALIBRE = Path("/Applications/calibre.app/Contents/MacOS")
_MAC_PREVIEWER = Path("/Applications/Kindle Previewer 4.app")

INSTALL_HELP = (
    "To build the Kindle (KFX) edition install: calibre (https://calibre-ebook.com), "
    "its 'KFX Output' plugin (calibre Preferences > Plugins > Get new plugins), and "
    "Kindle Previewer 4 (Amazon). For an EPUB-only build pass --kfx skip (or --kfx auto to skip only when the tools are missing)."
)


class KfxToolsMissing(RuntimeError):
    pass


@dataclass
class KfxResult:
    status: str                      # "built" | "skipped" | "failed"
    path: Path | None = None
    reason: str = ""
    log_path: Path | None = None
    removed_stale: bool = False
    checks: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def find_ebook_convert() -> Path | None:
    env = os.environ.get("FARSI2EPUB_EBOOK_CONVERT")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    found = shutil.which("ebook-convert")
    if found:
        return Path(found)
    mac = _MAC_CALIBRE / "ebook-convert"
    return mac if mac.is_file() else None


def _plugin_installed(ebook_convert: Path) -> bool:
    if os.environ.get("FARSI2EPUB_EBOOK_CONVERT"):
        return True                  # an explicit converter (tests, custom installs) is trusted
    customize = ebook_convert.with_name("calibre-customize")
    if not customize.is_file():
        return True                  # cannot check; the conversion itself will say
    try:
        out = subprocess.run([str(customize), "-l"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return True
    return "KFX Output" in out


def _previewer_present() -> bool:
    if os.environ.get("FARSI2EPUB_EBOOK_CONVERT"):
        return True
    if platform.system() == "Darwin":
        return _MAC_PREVIEWER.is_dir()
    return True                      # elsewhere the plugin locates it and reports if absent


def check_tools() -> Path:
    exe = find_ebook_convert()
    if exe is None:
        raise KfxToolsMissing("calibre's ebook-convert was not found. " + INSTALL_HELP)
    if not _plugin_installed(exe):
        raise KfxToolsMissing("calibre's 'KFX Output' plugin is not installed. " + INSTALL_HELP)
    if not _previewer_present():
        raise KfxToolsMissing(f"Kindle Previewer 4 was not found at {_MAC_PREVIEWER}. " + INSTALL_HELP)
    return exe


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def epub_fonts(epub_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(epub_path) as z:
        return {n: z.read(n) for n in z.namelist() if n.lower().endswith((".ttf", ".otf"))}


def verify_kfx(kfx_path: Path, epub_path: Path, log: str) -> tuple[bool, dict]:
    """The conditions under which the KFX was confirmed joined on the device."""
    data = kfx_path.read_bytes() if kfx_path.is_file() else b""
    fonts = epub_fonts(epub_path)
    meta = re.search(r"^Metadata: (.*)$", log, re.M)
    feats = re.search(r"^Features: (.*)$", log, re.M)
    checks = {
        "kfx_container": data[:4] == b"CONT",
        # Publisher Font on the Kindle is the EPUB's own font only if it is carried over intact.
        "epub_fonts_present": bool(fonts),
        "fonts_embedded_intact": bool(fonts) and all(b in data for b in fonts.values()),
        "language_fa": bool(meta) and re.search(r"\blanguage=fa\b", meta.group(1)) is not None,
        "persian_reflow_feature": bool(feats) and "fa-reflow-language" in feats.group(1),
        "previewer_converted": "Successful conversion to KPF" in log or "Successfully converted EPUB to KPF" in log,
    }
    return all(checks.values()), checks


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def kfx_path_for(epub_path: Path) -> Path:
    return epub_path.with_suffix(".kfx")


def build_kfx(epub_path: Path, mode: str = DEFAULT_MODE, log_dir: Path | None = None) -> KfxResult:
    """Convert `epub_path` to a KFX beside it. Never modifies the EPUB.

    mode: "auto" builds when the tools are present and skips with a reason otherwise;
    "require" treats missing tools as a failure; "skip" builds nothing.
    """
    if mode not in MODES:
        raise ValueError(f"unknown KFX mode {mode!r}")
    epub_path = Path(epub_path)
    final = kfx_path_for(epub_path)
    partial = final.with_name(".partial-" + final.name)
    log_dir = Path(log_dir) if log_dir else epub_path.parent
    log_path = log_dir / (final.stem + ".kfx.log")

    def finish(status: str, reason: str = "", checks: dict | None = None) -> KfxResult:
        stale = False
        if status != "built" and final.exists():
            final.unlink()
            stale = True
        if partial.exists():
            partial.unlink()
        return KfxResult(status, final if status == "built" else None, reason,
                         log_path if log_path.exists() else None, stale, checks or {})

    if mode == "skip":
        return finish("skipped", "--kfx skip")
    try:
        exe = check_tools()
    except KfxToolsMissing as exc:
        return finish("failed" if mode == "require" else "skipped", str(exc))

    # The command that produced the file confirmed on the device: defaults only.
    cmd = [str(exe), str(epub_path), str(partial)]
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CONVERT_TIMEOUT_S)
        log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log, rc = f"timed out after {CONVERT_TIMEOUT_S}s\n{exc.stdout or ''}", -1
    except OSError as exc:
        log, rc = f"failed to start: {exc}", -1
    log_path.write_text(log, encoding="utf-8")
    if rc != 0:
        return finish("failed", f"ebook-convert exited {rc}; see {log_path}")
    ok, checks = verify_kfx(partial, epub_path, log)
    if not ok:
        bad = ", ".join(k for k, v in checks.items() if not v)
        return finish("failed", f"KFX failed checks: {bad}; see {log_path}", checks)
    os.replace(partial, final)
    return KfxResult("built", final, "", log_path, False, checks)


def report(res: KfxResult) -> str:
    if res.status == "built":
        kb = res.path.stat().st_size / 1024
        return (f"  Kindle (KFX):    {res.path} ({kb:.1f} KB)\n"
                f"                   copy it to the Kindle's documents/ folder over USB "
                f"(Send to Kindle does not accept KFX)")
    line = f"  Kindle (KFX):    {res.status.upper()}: {res.reason}"
    if res.removed_stale:
        line += "\n                   (an older .kfx in out/ was removed so it cannot ship out of date)"
    return line
