"""Second reader for hard pages: Claude Opus on native-resolution bands.

Used by ``transcribe`` for a page that has no embedded-text oracle and still
trips the review gate after the normal pass (owner decisions D7, D16 and the
integration approval of 2026-11-14 in ``docs/bbox-old-fonts-plan.md``).

The request built here is the configuration that was measured, and nothing
else: arm ``t1r_opus`` of ``tests/transcription_experiments.py`` (same bands,
same instruction, same schema, same output limit and effort).
``tests/hard_reader_regression.py`` checks that equality; change neither side
alone. On the held-out typewritten pages it read at 2.5% word error (ignoring
joins and punctuation) where the normal pass was at 13%.

The reader returns printed lines, not Markdown. The words of the final page
come from this reader only; paragraph breaks and headings are carried over
from the first pass by aligning the two word sequences (``lines_to_markdown``).
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from . import config, llm
from .config import MODEL_HARD

READER_VERSION = 1
MAX_TOKENS = 12000
EFFORT = "medium"          # thinking cannot be disabled on Opus 5.5; effort is the control
BAND_OVERLAP_PX = 300
JPEG_ABOVE_BYTES = 3_700_000
UNREADABLE = "⟦؟⟧"
UNREADABLE_MD = "[؟]"

# ---------------------------------------------------------------------------
# Instruction (byte-identical to BANDS_READ_SYSTEM in the experiment script)
# ---------------------------------------------------------------------------

READ_RULES = """TRANSCRIPTION RULES (identify each typed word; write joins and spacing as printed)
- The page may be typewritten, photographed, blurred or stained, so letter shapes are crude: dots merge, break or vanish; the medial forms of ب پ ت ث ن ی can look alike; ه ع م and ف ق can blur together; a typewriter often leaves a small gap inside a word after a non-joining letter that is not a word break. Identify each word as the Persian word that was typed. Never write a non-word that merely copies a glyph defect.
- Use the surrounding words only to choose between letter readings that the visible shapes allow. Never write a word, name, number or phrase that the visible letters do not support, and never complete a sentence from what it "should" say.
- When you cannot identify a word from what is visible, write ⟦؟⟧ (one per unreadable word). When you are choosing between readings, write your best reading as ⟦word؟⟧. On a hard page, expect to use many markers: an unmarked wrong word is far worse than a marked right one. Mark each word you would not bet on.
- Word joins and splits exactly as typed: if the typist ran two words together with no space, write them together; if a prefix such as می or a suffix such as ها stands apart, write it apart. Never add a zero-width non-joiner (U+200C). Keep the spacing around punctuation as printed.
- Use standard Persian codepoints: always ی (U+06CC) and ک (U+06A9), never Arabic ي or ك. Persian digits (۰۱۲۳۴۵۶۷۸۹), in the order the typist typed them.
- Remove kashida/tatweel stretching.
- Do not transcribe signatures, stamps or seals, handwriting, or watermarks. Where stamp ink covers printed text, write the printed words you can read and mark the rest.
- Completeness: transcribe every printed line of body text, top to bottom, including headings. Never skip a line.

"""

BAND_FORMAT = """INPUT AND OUTPUT FORMAT (replaces any Markdown or field instructions)
- The page arrives as {n} overlapping horizontal bands of the same page, top to bottom, at full resolution, each preceded by a label "Band k:". Consecutive bands overlap by a few printed lines.
- Transcribe each band on its own and echo its number as band_index. For each band return `lines`: one string per printed line that is fully visible in that band, top to bottom, including lines that also appear in the neighbouring band. Skip only a line whose letters are cut by the band's top or bottom edge (it is complete in the neighbouring band).
- One string per physically printed line: never merge two printed lines or split one. Plain text: no Markdown."""


def system_prompt(n_bands: int) -> str:
    cur = llm.TRANSCRIBE_SYSTEM
    head, rest = cur.split("TRANSCRIPTION RULES", 1)
    _, rest = rest.split("IGNORE COMPLETELY", 1)
    ignore, _ = rest.split("MARKDOWN STRUCTURE", 1)
    rules = (head.split("You receive", 1)[0] + "You receive one page and write down its printed lines.\n\n"
             + READ_RULES + ("IGNORE COMPLETELY" + ignore).rstrip() + "\n")
    return rules + "\n" + BAND_FORMAT.format(n=n_bands)


class BandLines(BaseModel):
    band_index: int = Field(description="The k from the band's 'Band k:' label.")
    lines: list[str] = Field(description="Printed lines fully visible in this band, top to bottom.")


class BandsReading(BaseModel):
    bands: list[BandLines]


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

@dataclass
class BandImage:
    data: bytes
    media_type: str
    width: int
    height: int
    label: str = ""

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def band_rows(height: int, width: int, overlap: int = BAND_OVERLAP_PX) -> list[tuple[int, int]]:
    """Fewest equal-height bands, overlapping by `overlap` px, none of which
    the API downsizes (high-resolution tier)."""
    edge, cap = config.IMAGE_TIERS[0]
    cols = -(-width // config.IMAGE_PATCH_PX)
    max_h = min(edge, (cap // cols) * config.IMAGE_PATCH_PX)
    n = max(1, math.ceil((height - overlap) / (max_h - overlap)))
    bh = math.ceil((height + (n - 1) * overlap) / n)
    while -(-bh // config.IMAGE_PATCH_PX) * cols > cap or bh > edge:
        n += 1
        bh = math.ceil((height + (n - 1) * overlap) / n)
    step = (height - bh) / (n - 1) if n > 1 else 0
    rows = []
    for k in range(n):
        y0 = int(round(k * step))
        rows.append((y0, min(height, y0 + bh)))
    return rows


def band_source(pdf_path, page: int, hires_png: bytes):
    """(pixmap, kind): the page's embedded photo at native size when it is
    finer than the hi-res render, else the hi-res render."""
    import fitz
    doc = fitz.open(str(pdf_path))
    pg = doc[page - 1]
    render_zoom = config.LONG_EDGE_HI / max(pg.rect.width, pg.rect.height)
    best = None
    for info in pg.get_image_info(xrefs=True):
        bw = info["bbox"][2] - info["bbox"][0]
        if bw <= 0 or not info.get("xref"):
            continue
        ppt = info["width"] / bw
        if best is None or info["width"] * info["height"] > best[1]["width"] * best[1]["height"]:
            best = (ppt, info)
    if best and best[0] > render_zoom:
        pix = fitz.Pixmap(doc, best[1]["xref"])
        if pix.alpha or pix.n not in (1, 3):
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return pix, "embedded_native"
    return fitz.Pixmap(hires_png), "hires_render"


def band_images(pdf_path, page: int, hires_png: bytes) -> tuple[list[BandImage], dict]:
    import fitz
    import numpy as np
    pix, kind = band_source(pdf_path, page, hires_png)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    rows = band_rows(pix.height, pix.width)
    out = []
    for k, (y0, y1) in enumerate(rows):
        sub = np.ascontiguousarray(arr[y0:y1])
        bp = fitz.Pixmap(pix.colorspace, pix.width, y1 - y0, sub.tobytes(), False)
        data, mt = bp.tobytes("png"), "image/png"
        if len(data) > JPEG_ABOVE_BYTES:
            data, mt = bp.tobytes("jpg", jpg_quality=95), "image/jpeg"
        out.append(BandImage(data, mt, pix.width, y1 - y0, f"band{k + 1}:{y0}-{y1}"))
    return out, {"source": kind, "src_w": pix.width, "src_h": pix.height, "bands": len(rows)}


def build_request(pdf_path, page: int, page_count: int, title_hint: Optional[str], hires_png: bytes) -> dict:
    """The full API request as plain data (images still as BandImage)."""
    imgs, info = band_images(pdf_path, page, hires_png)
    hint = f"کتاب: {title_hint}" if title_hint else "عنوان کتاب نامشخص است."
    content: list = []
    for k, im in enumerate(imgs, 1):
        content += [f"Band {k}:", im]
    content.append(f"صفحهٔ {page} از {page_count}. {hint}\n"
                   f"Transcribe the {len(imgs)} bands of this page following the system instructions.")
    return {"model": MODEL_HARD, "max_tokens": MAX_TOKENS, "system": system_prompt(len(imgs)),
            "content": content, "schema": BandsReading,
            "kwargs": {"output_config": {"effort": EFFORT}}, "info": info}


def _image_block(img: BandImage) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": img.media_type,
                                        "data": base64.standard_b64encode(img.data).decode("ascii")}}


def read_page(client, pdf_path, page: int, page_count: int, title_hint: Optional[str],
              hires_png: bytes) -> tuple[list[str], dict, dict, float]:
    """One batched call. Returns (printed lines, info, usage, cost_usd)."""
    req = build_request(pdf_path, page, page_count, title_hint, hires_png)
    content = [_image_block(c) if isinstance(c, BandImage) else {"type": "text", "text": c}
               for c in req["content"]]

    def _call():
        response = client.messages.parse(
            model=req["model"], max_tokens=req["max_tokens"], system=req["system"],
            messages=[{"role": "user", "content": content}],
            output_format=req["schema"], **req["kwargs"])
        if response.parsed_output is None:
            raise RuntimeError(f"Model {req['model']} returned unparseable band output for page {page}")
        return response

    response = llm._call_with_retry(_call, page_no=page, what="hard-page read")
    bands = sorted(response.parsed_output.bands, key=lambda b: b.band_index)
    lines, warnings = stitch_bands([list(b.lines) for b in bands])
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    info = {**req["info"], "band_line_counts": [len(b.lines) for b in bands], "stitch_warnings": warnings}
    return lines, info, usage, llm.cost_of(response.usage, req["model"])


# ---------------------------------------------------------------------------
# Folding and band stitching (same rules as the experiment script)
# ---------------------------------------------------------------------------

ZWNJ = "‌"
_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک",
    **{chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)},
    "ـ": None, "​": None, "‍": None, "‎": None,
    "‏": None, "﻿": None, " ": " ",
})
_DIACRITICS = re.compile("[ً-ٰٟ]")
_PAGE_NO_LINE = re.compile(r"^[\s\-–—_()\d۰-۹٠-٩]*$")
_MARKER_RE = re.compile(r"⟦([^⟧]*)⟧")


def strict_fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_CHAR_MAP)


def ortho_fold_token(tok: str) -> str:
    tok = tok.replace(ZWNJ, "").replace("ۀ", "ه")
    tok = _DIACRITICS.sub("", tok)
    return "".join(ch for ch in tok if unicodedata.category(ch)[0] not in "PS")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def _line_key(s: str) -> str:
    return "".join(ortho_fold_token(t) for t in strict_fold(s).split())


def stitch_bands(bands: list[list[str]], sim_min: float = 0.6, window: int = 12) -> tuple[list[str], list[str]]:
    """Merge per-band line lists top to bottom. The tail of the merged text is
    aligned with the head of the next band (monotone, unmatched lines allowed
    on either side); a matched pair is kept once (the copy farther from its
    band edge), and an unmatched line inside the overlap is kept."""
    warnings: list[str] = []
    merged: list[str] = [l for l in (bands[0] if bands else []) if l.strip()]
    for bi, band in enumerate(bands[1:], 1):
        band = [l for l in band if l.strip()]
        if not band:
            warnings.append(f"band {bi} empty")
            continue
        t0 = max(0, len(merged) - window)
        A = merged[t0:]
        Bh = band[:window]
        ka, kb = [_line_key(x) for x in A], [_line_key(x) for x in Bh]
        na, nb = len(A), len(Bh)
        S = [[0.0] * (nb + 1) for _ in range(na + 1)]
        P = [[None] * (nb + 1) for _ in range(na + 1)]
        for i in range(na + 1):
            for j in range(nb + 1):
                if i == 0 and j == 0:
                    continue
                best, bp = -1e9, None
                if i > 0 and j > 0 and char_sim(ka[i - 1], kb[j - 1]) >= sim_min:
                    v = S[i - 1][j - 1] + 1
                    if v > best:
                        best, bp = v, "m"
                if i > 0:
                    v = S[i - 1][j] - (0.0 if j == 0 else 0.25)
                    if v > best:
                        best, bp = v, "a"
                if j > 0:
                    v = S[i][j - 1] - 0.25
                    if v > best:
                        best, bp = v, "b"
                S[i][j], P[i][j] = best, bp
        jbest = max(range(nb + 1), key=lambda j: (S[na][j], -j))
        pairs = []
        i, j = na, jbest
        while i > 0 or j > 0:
            bp = P[i][j]
            if bp == "m":
                pairs.append(("m", i - 1, j - 1)); i, j = i - 1, j - 1
            elif bp == "a":
                pairs.append(("a", i - 1, None)); i -= 1
            else:
                pairs.append(("b", None, j - 1)); j -= 1
        pairs.reverse()
        matched = [p for p in pairs if p[0] == "m"]
        if not matched:
            warnings.append(f"band {bi}: no overlap match; concatenated")
            merged = merged + band
            continue
        first_a = matched[0][1]
        region = [p for p in pairs if not (p[0] == "a" and p[1] < first_a)]
        out = merged[:t0 + first_a]
        nm = len(matched)
        mi = 0
        for kind, ia, jb in region:
            if kind == "m":
                out.append(A[ia] if mi < (nm + 1) // 2 else Bh[jb])
                mi += 1
            elif kind == "a":
                out.append(A[ia])
            else:
                out.append(Bh[jb])
        merged = out + band[jbest:]
    return merged, warnings


# ---------------------------------------------------------------------------
# Printed lines -> page Markdown
# ---------------------------------------------------------------------------

MIN_ALIGNED = 0.5      # share of the reader's words that must pair with a first-pass word
SNAP_WORDS = 2         # a carried-over break this close to a printed line start moves onto it
_HEADING = re.compile(r"^(#{1,3})\s+(.+)$")
_UNSUPPORTED = re.compile(r"^(```|\||\[\^\d+\]:)|\[\^\d+\]| --- ")


def plain_lines(lines: list[str]) -> tuple[list[str], list[str], int]:
    """Resolve the reader's markers: a marked guess keeps its word, an
    unreadable word becomes ``[؟]``; page-number-only lines are dropped.
    Returns (lines, guessed words, unreadable count)."""
    out, guessed, unread = [], [], 0
    for raw in lines:
        if _PAGE_NO_LINE.match(strict_fold(raw).strip()):
            continue

        def repl(m):
            nonlocal unread
            inner = m.group(1).strip().rstrip("؟?").strip()
            if not inner:
                unread += 1
                return f" {UNREADABLE_MD} "
            guessed.append(inner)
            return f" {inner} "

        line = " ".join(_MARKER_RE.sub(repl, raw).split())
        if line:
            out.append(line)
    return out, guessed, unread


def structure_blocks(first_md: str) -> Optional[list[tuple[str, str]]]:
    """[(kind, text)] of the first-pass Markdown, kind in h1/h2/h3/p, or None
    when it has structure this module cannot carry over (verse, tables,
    footnotes)."""
    blocks = []
    for line in first_md.split("\n"):
        s = line.strip()
        if not s:
            continue
        if _UNSUPPORTED.search(s):
            return None
        m = _HEADING.match(s)
        if m:
            blocks.append((f"h{len(m.group(1))}", m.group(2).strip()))
            continue
        s = re.sub(r"^>\s?", "", s)
        s = re.sub(r"^[-*+]\s+", "", s)
        s = s.replace("**", "").replace("__", "")
        if s:
            blocks.append(("p", s))
    return blocks


def _fold(word: str) -> str:
    return ortho_fold_token(strict_fold(word))


def _align(a: list[str], b: list[str], sim_min: float = 0.5) -> list[tuple[int, int]]:
    """Monotone pairs (i, j) of similar words, maximising total similarity."""
    na, nb = len(a), len(b)
    if not na or not nb:
        return []
    band = max(40, abs(na - nb) + max(na, nb) // 4)
    S = [[0.0] * (nb + 1) for _ in range(na + 1)]
    P = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        centre = i * nb / na
        row, up = S[i], S[i - 1]
        for j in range(1, nb + 1):
            best, bp = up[j], 1
            if row[j - 1] > best:
                best, bp = row[j - 1], 2
            if abs(j - centre) <= band:
                x, y = a[i - 1], b[j - 1]
                sim = 1.0 if x == y else (char_sim(x, y) if x and y else 0.0)
                if sim >= sim_min and up[j - 1] + sim > best:
                    best, bp = up[j - 1] + sim, 3
            row[j], P[i][j] = best, bp
    pairs = []
    i, j = na, nb
    while i > 0 and j > 0:
        bp = P[i][j]
        if bp == 3:
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif bp == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def lines_to_markdown(lines: list[str], first_md: str) -> tuple[str, list[str], dict]:
    """Page Markdown whose words are exactly the words of `lines` (markers
    already resolved), with block breaks and heading levels taken from the
    first-pass Markdown. Returns (text_md, headings, report).

    When too few words pair up, or the first pass has no usable structure,
    every printed line becomes its own paragraph (report["structure"] ==
    "printed_lines"), which the reviewer sees and can join.
    """
    words: list[str] = []
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(len(words))
        words.extend(line.split())
    blocks = structure_blocks(first_md) or []
    fwords: list[str] = []
    block_start: list[int] = []
    for _, text in blocks:
        block_start.append(len(fwords))
        fwords.extend(text.split())
    # punctuation-only words fold to "" and take no part in the alignment
    ai = [k for k, w in enumerate(fwords) if _fold(w)]
    bi = [k for k, w in enumerate(words) if _fold(w)]
    pairs = [(ai[i], bi[j]) for i, j in _align([_fold(fwords[k]) for k in ai], [_fold(words[k]) for k in bi])]
    aligned = len(pairs) / max(1, len(bi))
    report = {"aligned": round(aligned, 3), "structure": "first_pass"}
    if not words:
        return "", [], {**report, "structure": "empty"}
    if aligned < MIN_ALIGNED or not blocks:
        report["structure"] = "printed_lines"
        return "\n\n".join(lines) + "\n", [], report

    cuts = [0]
    pi = 0
    for b in range(1, len(blocks)):
        s = block_start[b]
        while pi < len(pairs) and pairs[pi][0] < s:
            pi += 1
        if pi == len(pairs):
            pos = len(words)
        else:
            i, j = pairs[pi]
            prev_j = pairs[pi - 1][1] if pi else -1
            pos = j - min(i - s, j - prev_j - 1)
        near = min(line_starts + [len(words)], key=lambda ls: abs(ls - pos))
        if abs(near - pos) <= SNAP_WORDS:
            pos = near
        cuts.append(max(pos, cuts[-1]))
    cuts.append(len(words))

    out: list[tuple[str, str]] = []
    for b, (kind, text) in enumerate(blocks):
        seg = words[cuts[b]:cuts[b + 1]]
        if not seg:
            continue
        if kind != "p" and len(seg) > 2 * len(text.split()) + 2:
            kind = "p"            # a heading must not swallow body text
        out.append((kind, " ".join(seg)))
    mark = {"h1": "# ", "h2": "## ", "h3": "### ", "p": ""}
    md = "\n\n".join(mark[k] + t for k, t in out) + "\n"
    return md, [t for k, t in out if k != "p"], report
