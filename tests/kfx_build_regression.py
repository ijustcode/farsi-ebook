"""Kindle (KFX) edition of `build`: regression test.

  ./venv/bin/python tests/kfx_build_regression.py                  # unit part + real conversion if the tools are installed
  ./venv/bin/python tests/kfx_build_regression.py --require-tools  # fail (not skip) when calibre / KFX Output / Kindle Previewer are missing

No API key, no network, no books/ needed (a synthetic Persian workspace in a temp dir).

The bug this guards (2026-11-20): on the owner's Kindle Paperwhite 2015 (firmware
5.16.2.1.1) the EPUB, delivered by Send to Kindle as AZW3, shows every Persian
letter isolated with "Publisher Font". The KFX built by calibre's KFX Output
plugin (Kindle Previewer) from the same EPUB, copied over USB, shows joined
letters in Vazirmatn with Publisher Font. The device check is manual; this test
pins the conditions of that confirmed file:
  K1 `build` produces <name>.kfx beside the EPUB (CLI, default mode).
  K2 the KFX is a KFX container that carries the EPUB's Vazirmatn files byte for byte,
     says language=fa and has Kindle Previewer's Persian reflow feature.
  K3 a failed or unverifiable conversion leaves no .kfx (no partial, no stale file),
     exits non-zero when the tools are present, and still leaves the EPUB.
  K4 missing tools: auto skips with an install hint, require fails, skip builds nothing.
It also guards the other consumers (iPad / Apple Books / any EPUB reader):
  E1 the KFX step never changes the EPUB (same bytes before and after, every mode).
  E2 the EPUB keeps its embedded Vazirmatn, @font-face, lang fa and RTL spine
     (nobody "fixes" the Kindle by stripping the EPUB's font).
  E3 `build --kfx skip` writes the same EPUB as before this feature (build_epub output).
  E4 epubcheck passes on the EPUB (when epubcheck is on PATH).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from click.testing import CliRunner  # noqa: E402

from farsi2epub import cli, epub as epub_mod, kindle  # noqa: E402
from farsi2epub.workspace import Workspace  # noqa: E402

FAILS: list[str] = []
SKIPS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


PAGES = [
    "# قصه‌ی نمونه\n\nمجید کتاب را برداشت و به کتابخانه رفت. بی‌بی گفت: «زود برگرد.»\n\n"
    "این متن برای آزمودن پیوستگی حروف فارسی است: گچ، ژاله، پرنده، کتاب‌ها، می‌روم.",
    "## بخش دوم\n\nفردا صبح زود بیدار شد و نان و پنیر خورد. **دوچرخه** را از انبار بیرون آورد.\n\n"
    "〃 〃 : سال ۱۳۶۴",
]


def make_ws(root: Path, slug: str = "kfx-synth") -> Workspace:
    ws = Workspace(slug, books_root=root)
    for d in (ws.root, ws.pages_dir, ws.hires_dir, ws.text_dir, ws.review_dir, ws.out_dir):
        d.mkdir(parents=True, exist_ok=True)
    ws.save_meta({"slug": slug, "title_fa": "قصه‌ی نمونه", "author_fa": "نویسنده‌ی آزمون", "title_en": slug})
    for i, text in enumerate(PAGES, start=1):
        ws.page_md_path(i).write_text(text, encoding="utf-8")
        ws.page_meta_path(i).write_text(json.dumps({"is_blank": False}), encoding="utf-8")
    return ws


def entries(epub_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(epub_path) as z:
        out = {n: z.read(n) for n in z.namelist()}
    # the only build-time-dependent value in an ebooklib EPUB
    for n in out:
        if n.endswith(".opf"):
            out[n] = re.sub(rb"<meta property=\"dcterms:modified\">[^<]*</meta>", b"", out[n])
    return out


FAKE = r'''#!{python}
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
mode = os.environ.get("FAKE_KFX_MODE", "ok")
with open(os.environ["FAKE_KFX_ARGS"], "w") as f:
    f.write("\n".join(sys.argv[1:]))
if mode == "exit1":
    print("conversion exploded"); sys.exit(1)
fonts = b""
if mode != "nofonts":
    with zipfile.ZipFile(src) as z:
        fonts = b"".join(z.read(n) for n in z.namelist() if n.endswith(".ttf"))
open(dst, "wb").write(b"CONT" + b"\0" * 16 + fonts)
print("Successfully converted EPUB to KPF")
print("Features: CanonicalFormat-1, " + ("" if mode == "nofa" else "fa-reflow-language-1, ") + "reflow-style-10")
print("Metadata: author=x, language=" + ("en" if mode == "nofa" else "fa") + ", title=x")
'''


def unit_part(tmp: Path) -> None:
    ws = make_ws(tmp / "books")
    out = epub_mod.build_epub(ws)
    before = sha(out)
    kfx = kindle.kfx_path_for(out)

    # E2: the EPUB the iPad reads keeps its font, CSS, language and direction.
    ent = entries(out)
    fonts = kindle.epub_fonts(out)
    css = b"".join(v for n, v in ent.items() if n.endswith(".css")).decode()
    opf = b"".join(v for n, v in ent.items() if n.endswith(".opf")).decode()
    asset = {p.name: p.read_bytes() for p in (ROOT / "assets" / "fonts").glob("Vazirmatn-*.ttf")}
    check("E2 EPUB embeds both Vazirmatn files unchanged",
          len(fonts) == 2 and sorted(fonts.values()) == sorted(asset.values()), f"{list(fonts)}")
    check("E2 EPUB CSS keeps @font-face Vazirmatn", "@font-face" in css and "Vazirmatn" in css)
    check("E2 EPUB language fa and RTL spine",
          "<dc:language>fa</dc:language>" in opf and 'page-progression-direction="rtl"' in opf)

    fake = tmp / "fake-ebook-convert"
    fake.write_text(FAKE.replace("{python}", sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    args_file = tmp / "args.txt"
    env_keep = dict(os.environ)
    os.environ["FARSI2EPUB_EBOOK_CONVERT"] = str(fake)
    os.environ["FAKE_KFX_ARGS"] = str(args_file)
    logs = ws.root / "kindle"
    try:
        os.environ["FAKE_KFX_MODE"] = "ok"
        r = kindle.build_kfx(out, mode="auto", log_dir=logs)
        check("K1 built with a working converter", r.status == "built" and kfx.is_file(), r.reason)
        check("K1 converter called with defaults only (the device-confirmed command)",
              args_file.read_text().splitlines() == [str(out), str(kfx.with_name(".partial-" + kfx.name))])
        check("K2 verification passes on a good file", all(r.checks.values()), str(r.checks))
        check("K1 log written outside out/", r.log_path is not None and r.log_path.parent == logs)
        check("E1 EPUB unchanged by a KFX build", sha(out) == before)

        for mode, key in (("nofonts", "fonts_embedded_intact"), ("nofa", "persian_reflow_feature"), ("exit1", None)):
            os.environ["FAKE_KFX_MODE"] = mode
            r = kindle.build_kfx(out, mode="auto", log_dir=logs)
            leftovers = [p.name for p in out.parent.iterdir() if p.suffix == ".kfx"]
            check(f"K3 {mode}: fails, no .kfx left (stale one removed)",
                  r.status == "failed" and not leftovers and r.removed_stale, f"{r.status} {leftovers}")
            if key:
                check(f"K3 {mode}: failing check named", r.checks.get(key) is False, str(r.checks))
            check(f"E1 {mode}: EPUB unchanged", sha(out) == before)
            os.environ["FAKE_KFX_MODE"] = "ok"
            kindle.build_kfx(out, mode="auto", log_dir=logs)        # restore a good KFX for the next stale check

        # CLI, default mode, with the converter failing: non-zero exit, EPUB still there.
        orig_load = cli._load_workspace
        cli._load_workspace = lambda slug: Workspace(slug, books_root=ws.books_root)
        try:
            os.environ["FAKE_KFX_MODE"] = "exit1"
            res = CliRunner().invoke(cli.main, ["build", ws.slug])
            check("K3 CLI: failed conversion exits non-zero, EPUB built, no .kfx",
                  res.exit_code != 0 and out.is_file() and not kfx.exists(), res.output[-300:])
            os.environ["FAKE_KFX_MODE"] = "ok"
            res = CliRunner().invoke(cli.main, ["build", ws.slug])
            check("K1 CLI default builds the .kfx", res.exit_code == 0 and kfx.is_file() and "Kindle (KFX)" in res.output,
                  res.output[-300:])
            # E3: --kfx skip writes the same EPUB as build_epub, and removes the KFX.
            ref = entries(out)
            res = CliRunner().invoke(cli.main, ["build", ws.slug, "--kfx", "skip"])
            check("E3 CLI --kfx skip: same EPUB as build_epub, no .kfx",
                  res.exit_code == 0 and entries(out) == ref and not kfx.exists(), res.output[-300:])
        finally:
            cli._load_workspace = orig_load
        before = sha(out)                     # the CLI runs rebuilt the EPUB (new dcterms:modified)

        # K4: tools missing.
        os.environ["FARSI2EPUB_EBOOK_CONVERT"] = str(tmp / "does-not-exist")
        r = kindle.build_kfx(out, mode="auto")
        check("K4 auto without tools: skipped with an install hint",
              r.status == "skipped" and "KFX Output" in r.reason and not kfx.exists(), r.reason)
        r = kindle.build_kfx(out, mode="require")
        check("K4 require without tools: failed", r.status == "failed")
        r = kindle.build_kfx(out, mode="skip")
        check("K4 skip: nothing built", r.status == "skipped" and not kfx.exists())
        check("E1 EPUB unchanged after every mode", sha(out) == before)
        # D29: the Kindle edition is part of every build by default.
        check("K5 default mode is require (D29)", kindle.DEFAULT_MODE == "require")
        cli._load_workspace = lambda slug: Workspace(slug, books_root=ws.books_root)
        try:
            out.unlink()
            res = CliRunner().invoke(cli.main, ["build", ws.slug])
            check("K5 CLI default without tools: exits non-zero, EPUB still written for the iPad",
                  res.exit_code != 0 and out.is_file() and "--kfx skip" in res.output, res.output[-300:])
            res = CliRunner().invoke(cli.main, ["build", ws.slug, "--kfx", "auto"])
            check("K5 CLI --kfx auto without tools: exits 0 with the EPUB", res.exit_code == 0 and out.is_file())
        finally:
            cli._load_workspace = orig_load
    finally:
        os.environ.clear()
        os.environ.update(env_keep)

    exe = shutil.which("epubcheck")
    if exe:
        p = subprocess.run([exe, str(out)], capture_output=True, text=True, timeout=300)
        check("E4 epubcheck passes", p.returncode == 0, p.stdout[-400:])
    else:
        SKIPS.append("E4 epubcheck not on PATH")


def real_part(tmp: Path, require: bool) -> None:
    os.environ.pop("FARSI2EPUB_EBOOK_CONVERT", None)
    try:
        kindle.check_tools()
    except kindle.KfxToolsMissing as exc:
        msg = f"real conversion: {exc}"
        if require:
            check("real conversion tools present", False, msg)
        else:
            SKIPS.append(msg)
        return
    ws = make_ws(tmp / "books-real")
    out = epub_mod.build_epub(ws)
    before = sha(out)
    print("running the real conversion (Kindle Previewer, about a minute)...")
    r = kindle.build_kfx(out, mode="require", log_dir=ws.root / "kindle")
    check("K1 real: KFX built", r.status == "built", r.reason)
    for k, v in r.checks.items():
        check(f"K2 real: {k}", v)
    check("E1 real: EPUB unchanged", sha(out) == before)
    check("K3 real: no partial file left", not any(p.name.startswith(".partial-") for p in out.parent.iterdir()))


def main() -> int:
    require = "--require-tools" in sys.argv
    with tempfile.TemporaryDirectory(prefix="kfx-regress-") as t:
        tmp = Path(t)
        unit_part(tmp)
        real_part(tmp, require)
    for s in SKIPS:
        print("SKIP", s)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}" + (f" ({len(SKIPS)} skipped)" if SKIPS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
