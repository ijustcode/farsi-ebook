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
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from . import config, llm
from .config import HARD_READERS, MODEL_HARD, gemini_price

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


def build_request(pdf_path, page: int, page_count: int, title_hint: Optional[str], hires_png: bytes,
                  reader: str = "opus") -> dict:
    """The full API request as plain data (images still as BandImage). The
    bands, instruction and schema are the same for every reader."""
    imgs, info = band_images(pdf_path, page, hires_png)
    hint = f"کتاب: {title_hint}" if title_hint else "عنوان کتاب نامشخص است."
    content: list = []
    for k, im in enumerate(imgs, 1):
        content += [f"Band {k}:", im]
    content.append(f"صفحهٔ {page} از {page_count}. {hint}\n"
                   f"Transcribe the {len(imgs)} bands of this page following the system instructions.")
    req = {"model": MODEL_HARD, "max_tokens": MAX_TOKENS, "system": system_prompt(len(imgs)),
           "content": content, "schema": BandsReading,
           "kwargs": {"output_config": {"effort": EFFORT}}, "info": info}
    if reader == "gemini":
        req.update(model=HARD_READERS["gemini"], max_tokens=GEMINI_MAX_TOKENS,
                   kwargs={"thinking_level": GEMINI_THINKING})
    elif reader != "opus":
        raise ValueError(f"unknown hard-page reader {reader!r}")
    return req


def _image_block(img: BandImage) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": img.media_type,
                                        "data": base64.standard_b64encode(img.data).decode("ascii")}}


# --- Gemini (Google Interactions API over stdlib HTTPS; no SDK) ---------------
# The request is arm `gg_carbon` of tests/transcription_experiments.py exactly
# (tests/hard_reader_regression.py holds the two together). The key is read from
# the environment at send time only and never stored, logged or put in a URL.

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
GEMINI_KEY_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
GEMINI_RESOLUTION = "ultra_high"
GEMINI_MAX_TOKENS = 10000
GEMINI_THINKING = "low"
GEMINI_ATTEMPTS = 4


class ReaderUnavailable(RuntimeError):
    """The chosen reader cannot be used at all (no key, access or billing refused)."""


def gemini_key() -> Optional[str]:
    import os
    llm.load_env()
    return next((os.environ[n] for n in GEMINI_KEY_ENV if os.environ.get(n)), None)


def _inline_schema(model: type) -> dict:
    """Strict JSON schema with $defs inlined (the API does not resolve $ref)."""
    schema = model.model_json_schema()

    def fix(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
    fix(schema)
    defs = schema.pop("$defs", {})

    def sub(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return sub(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: sub(v) for k, v in node.items()}
        if isinstance(node, list):
            return [sub(v) for v in node]
        return node
    return sub(schema)


def gemini_body(req: dict) -> dict:
    items = [{"type": "text", "text": req["system"]}]
    for c in req["content"]:
        if isinstance(c, BandImage):
            items.append({"type": "image", "data": base64.standard_b64encode(c.data).decode("ascii"),
                          "mime_type": c.media_type, "resolution": GEMINI_RESOLUTION})
        else:
            items.append({"type": "text", "text": c})
    return {"model": req["model"], "input": items,
            "response_format": {"type": "text", "mime_type": "application/json",
                                "schema": _inline_schema(req["schema"])},
            "generation_config": {"max_output_tokens": req["max_tokens"], **req["kwargs"]}}


def gemini_parse(data: dict) -> tuple[Optional[dict], dict]:
    """(parsed JSON or None, usage). Text arrives under steps[].content[]."""
    u = data.get("usage") or {}
    thought = int(u.get("total_thought_tokens") or 0)
    usage = {"input_tokens": int(u.get("total_input_tokens") or 0),
             "output_tokens": int(u.get("total_output_tokens") or 0) + thought}
    texts = [part["text"] for step in data.get("steps") or []
             if isinstance(step, dict) and step.get("type") == "model_output"
             for part in step.get("content") or []
             if isinstance(part, dict) and part.get("type") == "text" and "text" in part]
    try:
        return (json.loads(texts[-1]) if texts else None), usage
    except ValueError:
        return None, usage


def _gemini_post(body: dict, key: str, timeout: float = 600.0) -> tuple[int, dict]:
    import urllib.error
    import urllib.request
    request = urllib.request.Request(GEMINI_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"error": {"message": raw[:300]}}


def gemini_call(body: dict, schema: type, what: str):
    """POST one request and return (validated `schema` instance, usage, cost_usd).
    Busy or malformed answers are retried; a refused key, tier or quota raises
    ReaderUnavailable at once. A failure after answers were received carries
    their cost in `.cost` and `.usage`, since those attempts are billed."""
    import time
    key = gemini_key()
    if not key:
        raise ReaderUnavailable(f"no Gemini key ({' or '.join(GEMINI_KEY_ENV)} in the environment or .env)")
    price = gemini_price()
    spent, usage_total, last = 0.0, {"input_tokens": 0, "output_tokens": 0}, "no attempt"
    for attempt in range(GEMINI_ATTEMPTS):
        if attempt:
            time.sleep(min(30, 4 * 2 ** (attempt - 1)))
        try:
            status, data = _gemini_post(body, key)
        except OSError as exc:                       # transport failure
            last = f"{type(exc).__name__}"
            continue
        if status >= 400:
            msg = str((data.get("error") or {}).get("message", ""))[:200].replace(key, "[key]")
            last = f"http {status}: {msg}"
            if status in (401, 403) or "billing" in msg.lower() or "free tier" in msg.lower() or "quota" in msg.lower():
                raise ReaderUnavailable(f"Gemini refused the request ({last})")
            if status in (400, 404, 413, 422):
                break                                 # will not improve on retry
            continue
        parsed, usage = gemini_parse(data)
        for k in usage_total:
            usage_total[k] += usage[k]
        spent += usage["input_tokens"] / 1e6 * price["in"] + usage["output_tokens"] / 1e6 * price["out"]
        if parsed is not None:
            try:
                return schema.model_validate(parsed), usage_total, spent
            except Exception:                         # noqa: BLE001  wrong shape: try again
                last = "answer did not fit the schema"
                continue
        last = "no parseable answer"
    exc = RuntimeError(f"Gemini {what} failed: {last}")
    exc.cost, exc.usage = spent, usage_total       # failed attempts that returned are still billed
    raise exc


def gemini_items_body(model: str, items: list, schema: type, max_tokens: int, thinking_level: str = GEMINI_THINKING) -> dict:
    """Request body from text strings and (image bytes, mime type) pairs, in order."""
    out = []
    for c in items:
        if isinstance(c, str):
            out.append({"type": "text", "text": c})
        else:
            data, mime = c
            out.append({"type": "image", "data": base64.standard_b64encode(data).decode("ascii"),
                        "mime_type": mime, "resolution": GEMINI_RESOLUTION})
    return {"model": model, "input": out,
            "response_format": {"type": "text", "mime_type": "application/json", "schema": _inline_schema(schema)},
            "generation_config": {"max_output_tokens": max_tokens, "thinking_level": thinking_level}}


def _read_gemini(req: dict, page: int) -> tuple[BandsReading, dict, float]:
    return gemini_call(gemini_body(req), BandsReading, f"hard-page read for page {page}")


def read_page(client, pdf_path, page: int, page_count: int, title_hint: Optional[str],
              hires_png: bytes, reader: str = "opus") -> tuple[list[str], dict, dict, float]:
    """One batched call. Returns (printed lines, info, usage, cost_usd)."""
    req = build_request(pdf_path, page, page_count, title_hint, hires_png, reader)
    if reader == "gemini":
        reading, usage, cost = _read_gemini(req, page)
        bands = sorted(reading.bands, key=lambda b: b.band_index)
        lines, warnings = stitch_bands([list(b.lines) for b in bands])
        info = {**req["info"], "band_line_counts": [len(b.lines) for b in bands], "stitch_warnings": warnings}
        return lines, info, usage, cost
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


def _foot_zone(lines: list[str]) -> list[tuple[int, int, int]]:
    """Footnote definitions printed at the foot of the page, as (N, first
    line, end line): lines "N ـ text" with rising numbers that reach the end
    of the page, each with at most two continuation lines."""
    zone: list[tuple[int, int, int]] = []
    end, cont = len(lines), 0
    for li in range(len(lines) - 1, -1, -1):
        m = _FN_LINE.match(lines[li].translate(_ASCII_DIGITS))
        if m:
            n = int(m.group(1))
            if zone and n >= zone[0][0]:
                break
            zone.insert(0, (n, li, end))
            end, cont = li, 0
        else:
            cont += 1
            if cont > 2:
                break
    return zone


_FN_HEAD = re.compile(r"^[(\[]?[\d۰-۹٠-٩]{1,3}[)\]]?\s*[ـ\-–—=.:)]+\s*")
_MARK_TAIL = re.compile(r"^(.*[^\W\d_])(\d{1,2})$")      # a word with one or two digits printed against its end
FN_UNUSED, FN_MISREAD, FN_SKIP_ALONE, FN_SKIP_TAIL = 3.0, 1.2, 1.0, 2.0


def _reader_footnotes(lines: list[str]) -> tuple[list[str], list[tuple[int, int, int]], list[tuple[int, str]], set]:
    """Footnotes as the reader printed them. The definitions are the numbered
    lines at the foot of the page. The references are the small digits in the
    body, printed alone after a word or against its end; the reader misreads
    such digits often, so they are matched to the definitions in reading
    order (a repeated reference is allowed, a digit may be left as text when
    it fits nowhere), not by their value alone. The reader's numbering is the
    printed one, so these win over the first pass.

    Returns (body lines without those digits and without the definition
    lines, [(line index, index of the word that carries the reference, N)],
    [(N, text)], numbers that got at least one reference). Nothing is taken
    when no digit in the body refers to the foot lines."""
    zone = _foot_zone(lines)
    if not zone:
        return lines, [], [], set()
    nums = [n for n, _, _ in zone]
    body = [l.split() for l in lines[:zone[0][1]]]
    cands = []                      # (line, token, value, alone)
    for li, toks in enumerate(body):
        for k, t in enumerate(toks):
            v = _num(t)
            if v is not None and k and v < 100:
                cands.append((li, k, v, True))
                continue
            m = _MARK_TAIL.match(t.translate(_ASCII_DIGITS))
            if m:
                cands.append((li, k, int(m.group(2)), False))
    # f[i][j]: least cost after i candidates with definition j the last one referred to (j = 0: none yet)
    INF = float("inf")
    nc, nd = len(cands), len(nums)
    f = [[INF] * (nd + 1) for _ in range(nc + 1)]
    back = [[None] * (nd + 1) for _ in range(nc + 1)]
    f[0][0] = 0.0
    for i, (_, _, v, alone) in enumerate(cands):
        for j in range(nd + 1):
            if f[i][j] == INF:
                continue
            c = f[i][j] + (FN_SKIP_ALONE if alone else FN_SKIP_TAIL)
            if c < f[i + 1][j]:
                f[i + 1][j], back[i + 1][j] = c, (j, None)
            for j2 in range(max(j, 1), nd + 1):
                c = f[i][j] + FN_UNUSED * max(0, j2 - j - 1) + (0.0 if nums[j2 - 1] == v else FN_MISREAD)
                if c < f[i + 1][j2]:
                    f[i + 1][j2], back[i + 1][j2] = c, (j, j2)
    j = min(range(nd + 1), key=lambda q: f[nc][q] + FN_UNUSED * (nd - q))
    assigned = [None] * nc
    for i in range(nc, 0, -1):
        j, took = back[i][j][0], back[i][j][1]
        assigned[i - 1] = took
    if not any(assigned):
        return lines, [], [], set()     # numbered lines with no reference digit in the body: a list, not footnotes
    marks, gone = [], {}
    for (li, k, _, alone), took in zip(cands, assigned):
        if took is None:
            continue
        n = nums[took - 1]
        shift = gone.get(li, 0)
        if alone:
            del body[li][k - shift]
            gone[li] = shift + 1
            marks.append((li, k - shift - 1, n))
        else:
            tok = body[li][k - shift]
            body[li][k - shift] = tok[:len(_MARK_TAIL.match(tok.translate(_ASCII_DIGITS)).group(1))]
            marks.append((li, k - shift, n))
    defs = [(n, _FN_HEAD.sub("", " ".join(" ".join(lines[a:e]).split()))) for n, a, e in zone]
    return [" ".join(t) for t in body], marks, defs, {n for _, _, n in marks}


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
    lines, marks, reader_defs, done = _reader_footnotes(lines)
    # What the reader settled is taken out of the first pass's structure: its
    # definitions always, and a reference unless the reader printed no digit
    # for that footnote (then the first pass's place for it is used).
    settled = {n for n, _ in reader_defs}
    kept = [b for b, (k, _) in enumerate(blocks) if not (k.startswith("fn") and int(k[2:]) in settled)]
    remap = {b: i for i, b in enumerate(kept)}
    blocks = [blocks[b] for b in kept]
    refs = [(remap[b], k, n) for b, k, n in refs if n not in done and b in remap]
    first_defs = {int(k[2:]) for k, _ in blocks if k.startswith("fn")} | settled
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
            report["footnotes"] = {"refs": len(marks), "defs": len(reader_defs), "refs_without_digit": 0,
                                   "refs_dropped": 0, "defs_from_first_pass": 0, "from_reader": len(reader_defs)}
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
    foot = {n: line_starts[a] for n, a, _ in _foot_zone(lines)}
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
    unmatched = dropped = 0
    pair_i = [i for i, _ in pairs]
    for b, k, n in refs:
        lo, hi = cuts[b], cuts[b + 1]
        if hi <= lo:
            dropped += 1
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
            elif n in first_defs:
                unmatched += 1      # the reader printed no digit there; the first pass's reference is kept
            else:
                dropped += 1        # no digit and no definition anywhere: not a footnote
                continue
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
        report["footnotes"] = {"refs": len(refs) - dropped + len(marks), "defs": len(defs),
                               "refs_without_digit": unmatched, "refs_dropped": dropped,
                               "defs_from_first_pass": defs_first, "from_reader": len(reader_defs)}
    mark = {"h1": "# ", "h2": "## ", "h3": "### ", "p": ""}
    md = with_defs("\n\n".join(mark[k] + t for k, t in out) + "\n", defs)
    return md, [_FN_REF.sub("", t).strip() for k, t in out if k != "p"], report
