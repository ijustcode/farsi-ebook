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
import bisect
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
_UNSUPPORTED = re.compile(r"^(```|\|)| --- ")
_FN_DEF = re.compile(r"^\[\^(\d+)\]:\s*(.*)$")
_FN_REF = re.compile(r"\[\^(\d+)\]")
_ASCII_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUM_TOKEN = re.compile(r"^[(\[]?(\d{1,3})[)\]]?$")
_NUM_TAIL = re.compile(r"^(.*[^\d(\[])[(\[]?(\d{1,3})[)\]]?$")
_FN_LINE = re.compile(r"^[(\[]?(\d{1,3})[)\]]?\s*[ـ\-–—=.:)]+\s*(\S.*)$")
_FN_LEAD = set("ـ-–—=.:)")


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


def _num(token: str) -> Optional[int]:
    """The number a free-standing digit token stands for (Persian or ASCII,
    optionally bracketed), else None."""
    m = _NUM_TOKEN.match(token.translate(_ASCII_DIGITS))
    return int(m.group(1)) if m else None


def structure_blocks(first_md: str, refs: Optional[list] = None) -> Optional[list[tuple[str, str]]]:
    """[(kind, text)] of the first-pass Markdown, kind in h1/h2/h3/p or fnN (the
    definition of footnote N), or None when it has structure this module cannot
    carry over (verse, tables). Footnote references are taken out of the text;
    with `refs` given, each is appended as (block index, index of the word it
    follows inside the block, N)."""
    blocks: list[tuple[str, str]] = []

    def add(kind: str, text: str) -> None:
        words = []
        for tok in _FN_REF.sub(lambda m: f" \x00{m.group(1)} ", text).split():
            if tok.startswith("\x00"):
                if refs is not None:
                    refs.append((len(blocks), len(words) - 1, int(tok[1:])))
            else:
                words.append(tok)
        if words:
            blocks.append((kind, " ".join(words)))

    for line in first_md.split("\n"):
        s = line.strip()
        if not s:
            continue
        if _UNSUPPORTED.search(s):
            return None
        m = _FN_DEF.match(s)
        if m:
            blocks.append((f"fn{int(m.group(1))}", " ".join(_FN_REF.sub(" ", m.group(2)).split())))
            continue
        m = _HEADING.match(s)
        if m:
            add(f"h{len(m.group(1))}", m.group(2).strip())
            continue
        s = re.sub(r"^>\s?", "", s)
        s = re.sub(r"^[-*+]\s+", "", s)
        add("p", s.replace("**", "").replace("__", ""))
    return blocks


def _reader_footnotes(lines: list[str], known: set) -> tuple[list[str], list[tuple[int, int, int]], list[tuple[int, str]]]:
    """Footnotes the first pass did not report but the reader printed: bottom
    lines of the form "N ـ text" whose number also stands alone after a word
    in the body. Returns (lines without those tokens and lines,
    [(line index, index of the word the reference follows, N)], [(N, text)])."""
    first = len(lines)
    while first > 0 and _FN_LINE.match(lines[first - 1].translate(_ASCII_DIGITS)):
        first -= 1
    body = [l.split() for l in lines[:first]]
    marks, defs, keep = [], [], []
    for line in lines[first:]:
        n = int(_FN_LINE.match(line.translate(_ASCII_DIGITS)).group(1))
        spot = None if n in known else next(((li, k) for li, toks in enumerate(body)
                                            for k, t in enumerate(toks) if k and _num(t) == n), None)
        if spot is None:
            keep.append(line)
            continue
        li, k = spot
        del body[li][k]
        marks.append((li, k - 1, n))
        toks = line.split()
        del toks[0]
        while toks and all(c in _FN_LEAD for c in toks[0]):
            del toks[0]
        defs.append((n, " ".join(toks)))
    return [" ".join(t) for t in body] + keep, marks, defs


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
    refs: list[tuple[int, int, int]] = []
    blocks = structure_blocks(first_md, refs) or []
    known = {int(k[2:]) for k, _ in blocks if k.startswith("fn")} | {n for _, _, n in refs}
    lines, marks, reader_defs = _reader_footnotes(lines, known)
    words: list[str] = []
    line_starts: list[int] = []
    suffix: dict[int, str] = {}
    for li, line in enumerate(lines):
        for _, k, n in (m for m in marks if m[0] == li):
            suffix[len(words) + k] = suffix.get(len(words) + k, "") + f"[^{n}]"
        line_starts.append(len(words))
        words.extend(line.split())
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
    def with_defs(md: str, defs: list[tuple[int, str]]) -> str:
        return md + ("\n" + "".join(f"[^{n}]: {t}\n" for n, t in defs) if defs else "")

    if aligned < MIN_ALIGNED or not blocks:
        report["structure"] = "printed_lines"
        k = 0
        out_lines = []
        for line in lines:
            toks = [w + suffix.get(k + i, "") for i, w in enumerate(line.split())]
            k += len(toks)
            if toks:
                out_lines.append(" ".join(toks))
        if reader_defs:
            report["footnotes"] = {"refs": len(reader_defs), "defs": len(reader_defs), "from_reader": len(reader_defs)}
        return with_defs("\n\n".join(out_lines) + "\n", reader_defs), [], report

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

    # A footnote definition starts where the reader printed its "N ـ" line at
    # the foot of the page: that is firmer than word similarity for a
    # one-word definition.
    foot: dict[int, int] = {}
    for li in range(len(lines) - 1, -1, -1):
        m = _FN_LINE.match(lines[li].translate(_ASCII_DIGITS))
        if not m:
            break
        foot[int(m.group(1))] = line_starts[li]
    fixed, last = {}, -1
    for b, (kind, _) in enumerate(blocks):
        if kind.startswith("fn") and foot.get(int(kind[2:]), -1) > last:
            fixed[b] = last = foot[int(kind[2:])]
    if fixed:
        for b in range(1, len(blocks)):
            cuts[b] = fixed[b] if b in fixed else max(cuts[b], cuts[b - 1])
        for b in range(len(blocks) - 1, 0, -1):
            if b not in fixed:
                cuts[b] = min(cuts[b], cuts[b + 1])

    # Footnote references the first pass saw: the reader prints the number as
    # a digit after the word; that digit becomes the [^N] of the word before it.
    deleted: set[int] = set()
    unmatched = 0
    pair_i = [i for i, _ in pairs]
    for b, k, n in refs:
        lo, hi = cuts[b], cuts[b + 1]
        if hi <= lo:
            unmatched += 1
            continue
        g = block_start[b] + max(k, 0)
        at = bisect.bisect_right(pair_i, g) - 1
        if at < 0:
            p = lo
        else:
            i, j = pairs[at]
            nxt = pairs[at + 1][1] if at + 1 < len(pairs) else len(words)
            p = min(j + (g - i), max(j, nxt - 1))
        p = min(max(p, lo), hi - 1)
        found = next((q for q in range(max(p, lo + 1), min(hi, p + 4)) if q not in deleted and _num(words[q]) == n), None)
        if found is not None:
            deleted.add(found)
            p = found - 1
            while p > lo and p in deleted:
                p -= 1
        else:
            tail = next((q for q in range(p, min(hi, p + 2))
                         if (m := _NUM_TAIL.match(words[q].translate(_ASCII_DIGITS))) and int(m.group(2)) == n), None)
            if tail is not None:
                p = tail
                words[p] = words[p][:len(_NUM_TAIL.match(words[p].translate(_ASCII_DIGITS)).group(1))]
            else:
                unmatched += 1      # the reader printed no digit there; the first pass's reference is kept
        suffix[p] = suffix.get(p, "") + f"[^{n}]"

    out: list[tuple[str, str]] = []
    defs: list[tuple[int, str]] = []
    defs_first = 0
    for b, (kind, text) in enumerate(blocks):
        seg = [words[q] + suffix.get(q, "") for q in range(cuts[b], cuts[b + 1]) if q not in deleted]
        if kind.startswith("fn"):
            n = int(kind[2:])
            for _ in range(3):
                if seg and (_num(seg[0]) == n or all(c in _FN_LEAD for c in seg[0])):
                    del seg[0]
            if not seg:
                seg, defs_first = text.split(), defs_first + 1   # never leave a reference without its definition
            defs.append((n, " ".join(seg)))
            continue
        if not seg:
            continue
        if kind != "p" and len(seg) > 2 * len(text.split()) + 2:
            kind = "p"            # a heading must not swallow body text
        out.append((kind, " ".join(seg)))
    defs += reader_defs
    if defs or refs:
        report["footnotes"] = {"refs": len(refs) + len(reader_defs), "defs": len(defs), "refs_without_digit": unmatched,
                               "defs_from_first_pass": defs_first, "from_reader": len(reader_defs)}
    mark = {"h1": "# ", "h2": "## ", "h3": "### ", "p": ""}
    md = with_defs("\n\n".join(mark[k] + t for k, t in out) + "\n", defs)
    return md, [_FN_REF.sub("", t).strip() for k, t in out if k != "p"], report
