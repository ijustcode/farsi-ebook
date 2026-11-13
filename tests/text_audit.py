#!/usr/bin/env python3
"""Text audit: a local browser tool in which the owner corrects model-drafted
transcriptions of photographed typewritten pages (hadpi-articles p3-p5).

Evaluation tooling only; production code never imports ``tests/``. No API
calls, no network beyond 127.0.0.1, no new dependencies (numpy + PyMuPDF).

    ./venv/bin/python tests/text_audit.py serve [--port 8771] [--open] [--root DIR]
    ./venv/bin/python tests/text_audit.py status [--root DIR]
    ./venv/bin/python tests/text_audit.py prepare --page N [--root DIR]   # page image + line geometry only

Files, all under ``<root>/books/hadpi-articles/truth/`` and only for pages 3-5:

    000N.lines.draft.txt        the original draft, byte for byte, written once, never rewritten
    000N.text_audit.json        the tool's state (lines, markers, choices, crop association, status)
    000N.text_audit.undo.jsonl  append-only undo log (one record per action and per undo)
    000N.lines.txt              the export, rewritten atomically after every action
    000N.page.png               page image used for display (only when rendered here)

The export is "not finished" until the state file says ``status: "audited"``
(every line accepted and no unresolved marker); ``export.final`` repeats it.
Until then the export still contains the draft's unresolved markers. In a
final export the only marker that can appear is ``⟦؟⟧`` (a word the owner
declared unreadable).

Nothing is ever normalised: spaces, ZWNJ, word joins and punctuation spacing
come out exactly as chosen or typed.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TOOL_VERSION = "text_audit/1"
SCHEMA = "farsi2epub.text-audit/v1"
SLUG = "hadpi-articles"
ALLOWED_PAGES = (3, 4, 5)
DEFAULT_PORT = 8771
FONT = REPO / "assets" / "fonts" / "Vazirmatn-Regular.ttf"
SENT = "\x00"            # placeholder for "nothing here" while rendering
OPEN, CLOSE, Q, BRK_INNER = "⟦", "⟧", "؟", "↵؟"
UNREADABLE = "⟦؟⟧"
GAP_ABSENT, GAP_UNREAD = "∅", "…"
CROP_PAD = 0.35          # same padding as the existing crops
CTX_EXT = 0.4            # context rows above/below the crop band, in units of the line band height


class AuditError(Exception):
    pass


class Conflict(Exception):
    def __init__(self, doc):
        super().__init__("this page was changed in another tab; it has been reloaded")
        self.doc = doc


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(Path(p).read_bytes())


def canonical(v) -> bytes:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, v) -> None:
    with path.open("ab") as f:
        f.write(canonical(v) + b"\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Line model: parse, render
# ---------------------------------------------------------------------------
class Ids:
    def __init__(self, nm=1, nl=1):
        self.nm, self.nl = nm, nl

    def mark(self):
        self.nm += 1
        return self.nm - 1

    def line(self):
        self.nl += 1
        return self.nl - 1


def merge_text(segs: list) -> list:
    out = []
    for s in segs:
        if s["t"] == "text":
            if not s["v"]:
                continue
            if out and out[-1]["t"] == "text":
                out[-1] = {"t": "text", "v": out[-1]["v"] + s["v"]}
                continue
            out.append({"t": "text", "v": s["v"]})
        else:
            out.append(s)
    return out


def parse_text(text: str, ids: Ids, resolve_unreadable: bool = False, where: str = "") -> list:
    """Text with markers -> segments. ``⟦؟⟧`` stays an unresolved marker when it
    comes from a draft, and becomes an already-resolved 'unreadable' marker when
    ``resolve_unreadable`` (text the owner typed)."""
    if SENT in text or "\n" in text or "\r" in text:
        raise AuditError(f"{where}text may not contain line breaks or NUL")
    segs, pos = [], 0
    for m in re.finditer("⟦([^⟧]*)⟧", text):
        pre = text[pos:m.start()]
        if OPEN in pre or CLOSE in pre:
            raise AuditError(f"{where}unbalanced marker bracket")
        segs.append({"t": "text", "v": pre})
        inner = m.group(1)
        if OPEN in inner:
            raise AuditError(f"{where}nested marker")
        if inner == BRK_INNER:
            segs.append({"t": "brk", "id": ids.mark()})
        else:
            if not inner.endswith(Q):
                raise AuditError(f"{where}marker without closing ؟: {m.group(0)[:20]}")
            body = inner[:-1]
            alts = [] if body == "" else body.split("|")
            seg = {"t": "mark", "id": ids.mark(), "alts": alts, "choice": None}
            if resolve_unreadable:
                if alts:
                    pass            # a typed marker with alternatives stays unresolved
                else:
                    seg["choice"] = {"k": "unreadable"}
            segs.append(seg)
        pos = m.end()
    tail = text[pos:]
    if OPEN in tail or CLOSE in tail:
        raise AuditError(f"{where}unbalanced marker bracket")
    segs.append({"t": "text", "v": tail})
    return merge_text(segs)


def raw_mark(seg: dict) -> str:
    if seg["t"] == "brk":
        return OPEN + BRK_INNER + CLOSE
    return OPEN + "|".join(seg["alts"]) + Q + CLOSE


def mark_value(seg: dict):
    """Resolved text of a marker segment (SENT for 'nothing'), or None if unresolved."""
    c = seg.get("choice")
    if c is None:
        return None
    k = c["k"]
    if k == "alt":
        a = seg["alts"][c["i"]]
        if a in (GAP_ABSENT, ""):
            return SENT
        if a == GAP_UNREAD:
            return UNREADABLE
        return a
    if k == "custom":
        return c["v"] if c["v"] else SENT
    if k == "unreadable":
        return UNREADABLE
    if k == "empty":
        return SENT
    raise AuditError(f"bad choice {c!r}")


def render_segs(segs: list) -> str:
    parts = []
    for s in segs:
        if s["t"] == "text":
            parts.append(s["v"])
        else:
            v = mark_value(s) if s["t"] == "mark" else None
            parts.append(raw_mark(s) if v is None else v)
    s = "".join(parts)
    s = re.sub(SENT + "+", SENT, s)
    s = re.sub("(?<= )" + SENT + " ", "", s)
    s = re.sub("^" + SENT + " ", "", s)
    s = re.sub(" " + SENT + "$", "", s)
    return s.replace(SENT, "")


def unresolved_count(line: dict) -> int:
    return sum(1 for s in line["segs"]
               if s["t"] == "brk" or (s["t"] == "mark" and s["choice"] is None))


def seg_len(s: dict) -> int:
    if s["t"] == "text":
        return len(s["v"])
    if s["t"] == "brk":
        return 0
    v = mark_value(s)
    if v is not None:
        return 0 if v == SENT else len(v)
    alts = [a for a in s["alts"] if a not in (GAP_ABSENT, GAP_UNREAD, "")]
    return len(alts[0]) if alts else 4


def mark_positions(line: dict) -> dict:
    """Approximate fraction (0 = start of reading, 1 = end) at the middle of each marker."""
    total = sum(seg_len(s) for s in line["segs"]) or 1
    out, cum = {}, 0
    for s in line["segs"]:
        n = seg_len(s)
        if s["t"] != "text":
            out[s["id"]] = round((cum + n / 2) / total, 4)
        cum += n
    return out


def invisible_note(alts: list) -> str:
    """One short line saying how alternatives differ when it is only spaces/ZWNJ."""
    def base(a):
        return re.sub("[ ‌ ‏‎]", "", a)
    groups: dict = {}
    for i, a in enumerate(alts):
        groups.setdefault(base(a), []).append(i + 1)
    notes = []
    for idxs in groups.values():
        if len(idxs) > 1 and len({alts[i - 1] for i in idxs}) > 1:
            notes.append(", ".join(map(str, idxs)))
    if not notes:
        return ""
    return ("Same letters, different spacing: " + "; ".join("alternatives " + n for n in notes) +
            " (differences are in spaces / ZWNJ / word joins; look at how it was typed).")


# ---------------------------------------------------------------------------
# Break pairs
# ---------------------------------------------------------------------------
def break_pairs(lines: list) -> dict:
    """line index -> (first, second_or_None) for lines that begin with a disputed-break marker."""
    out, i, n = {}, 0, len(lines)
    def has(j):
        return any(s["t"] == "brk" for s in lines[j]["segs"])
    while i < n:
        if has(i):
            if i + 1 < n and has(i + 1):
                out[i] = out[i + 1] = (i, i + 1)
                i += 2
                continue
            out[i] = (i, None)
        i += 1
    return out


# ---------------------------------------------------------------------------
# Paths, state creation
# ---------------------------------------------------------------------------
def truth_dir(root: Path) -> Path:
    return Path(root) / "books" / SLUG / "truth"


def paths_for(root: Path, page: int) -> dict:
    t = truth_dir(root)
    return dict(lines=t / f"{page:04d}.lines.txt", draft=t / f"{page:04d}.lines.draft.txt",
                state=t / f"{page:04d}.text_audit.json", undo=t / f"{page:04d}.text_audit.undo.jsonl",
                page_png=t / f"{page:04d}.page.png", crops=t / "crops")


def check_page(page) -> int:
    if page not in ALLOWED_PAGES:
        raise AuditError(f"page {page!r} is not one of the pages this tool may write ({', '.join(map(str, ALLOWED_PAGES))})")
    return int(page)


def new_line(ids: Ids, segs: list, crop=None, crop_src="auto", state="open") -> dict:
    return dict(id=ids.line(), segs=merge_text(segs), state=state, crop=crop, crop_src=crop_src)


def expected_chars(segs: list) -> int:
    """Characters a printed line is expected to hold: plain text, and the first real reading of each marker."""
    n = 0
    for s in segs:
        if s["t"] == "text":
            n += len(s["v"])
        elif s["t"] == "mark":
            alts = [a for a in s["alts"] if a not in (GAP_ABSENT, GAP_UNREAD, "")]
            n += len(alts[0]) if alts else 4
    return max(1, n)


def plain_first(segs: list) -> str:
    out = []
    for s in segs:
        if s["t"] == "text":
            out.append(s["v"])
        elif s["t"] == "mark":
            alts = [a for a in s["alts"] if a not in (GAP_ABSENT, GAP_UNREAD, "")]
            out.append(alts[0] if alts else "")
    return "".join(out)


PAGENO_RE = re.compile(r"^[\s\-\u2013\u2014]*[0-9\u06f0-\u06f9\u0660-\u0669]{1,4}[\s\-\u2013\u2014]*$")


def auto_assign(crops: list, exp: list, texts: list | None = None) -> dict:
    """Monotone assignment of crops (top to bottom) to draft lines (1-based among non-empty lines) by
    robust ink width against expected character count, with cheap skips for isolated or unusually
    tall strips (page number, stamps, handwriting) and for strips after the last line."""
    import math
    import statistics
    n, m = len(exp), len(crops)
    if not n or not m:
        return {}
    W = [max(1.0, float(c.get("ink_w") or (c["x1"] - c["x0"]))) for c in crops]
    H = [c["up"] + c["dn"] for c in crops]
    Y = [sum(c["ys"]) / len(c["ys"]) for c in crops]
    medH = statistics.median(H)
    gaps = [Y[j + 1] - Y[j] for j in range(m - 1)]
    spacing = statistics.median(gaps) if gaps else 1.0
    skip = []
    for j in range(m):
        near = min([g for g in (gaps[j - 1] if j else None, gaps[j] if j < m - 1 else None) if g is not None] or [spacing])
        s = 0.8
        if near > 1.5 * spacing:
            s = 0.35                  # isolated strip
        elif W[j] >= 0.5 * statistics.median(sorted(W)[m // 2:]):
            s = 2.0                   # a long strip is probably a text line: keep it unless nothing fits
        elif H[j] > 1.5 * medH:
            s = 0.4                   # unusually tall and not long: stamp, signature, handwriting
        skip.append(s)
    SKIP_LINE = 2.0
    # a small isolated strip at the top only suits a page-number line; a strip much thinner than a text line (underline) suits nothing
    full = statistics.median(sorted(W)[m // 2:])
    pn_crop = [j == 0 and skip[j] <= 0.4 and W[j] < 0.25 * full for j in range(m)]
    weak = [H[j] < 0.6 * medH for j in range(m)]          # much thinner than a text line: an underline, a smudge
    for j in range(m):
        if pn_crop[j] or weak[j]:
            skip[j] = min(skip[j], 0.35)
    pn_line = [bool(texts and PAGENO_RE.match(texts[i])) for i in range(n)]
    pairs = sorted(zip(exp, W), key=lambda x: -x[0])[: max(3, n // 2)]
    ratio = statistics.median([w / e for e, w in pairs])
    assign = {}
    for _ in range(3):
        INF = 1e18
        D = [[INF] * (m + 1) for _ in range(n + 1)]
        P = [[None] * (m + 1) for _ in range(n + 1)]
        D[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                d = D[i][j]
                if d >= INF:
                    continue
                if i < n and j < m:
                    c = min(1.5, abs(math.log(W[j] / max(1.0, ratio * exp[i]))))
                    if weak[j] or (pn_crop[j] and not pn_line[i]):
                        c += 1.5
                    if d + c < D[i + 1][j + 1]:
                        D[i + 1][j + 1], P[i + 1][j + 1] = d + c, (i, j, "m")
                if j < m:
                    sc = min(skip[j], 0.6) if i == n else skip[j]
                    if d + sc < D[i][j + 1]:
                        D[i][j + 1], P[i][j + 1] = d + sc, (i, j, "c")
                if i < n and d + SKIP_LINE < D[i + 1][j]:
                    D[i + 1][j], P[i + 1][j] = d + SKIP_LINE, (i, j, "l")
        assign, i, j = {}, n, m
        while i or j:
            pi, pj, k = P[i][j]
            if k == "m":
                assign[pi + 1] = crops[pj]["id"]
            i, j = pi, pj
        got = [(exp[i - 1], W[next(k for k, c in enumerate(crops) if c["id"] == cid)]) for i, cid in assign.items()]
        if got:
            ratio = statistics.median([w / e for e, w in got])
    return assign


def build_state(page: int, draft_bytes: bytes, crops: list, image: dict, auto_files: dict | None = None) -> dict:
    """New state from the draft. ``crops`` is the inventory (top to bottom);
    ``auto_files`` maps a draft line number (1-based among non-empty lines) to a crop id."""
    page = check_page(page)
    try:
        text = draft_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"draft is not UTF-8: {exc}")
    if "\r" in text:
        raise AuditError("draft contains carriage returns; refusing")
    final_nl = text.endswith("\n")
    raw_lines = text.split("\n")
    if final_nl:
        raw_lines = raw_lines[:-1]
    ids = Ids()
    parsed = [parse_text(raw, ids, where=f"line {n}: ") for n, raw in enumerate(raw_lines, 1)]
    ids = Ids(ids.nm, 1)
    nonblank = [i for i, raw in enumerate(raw_lines) if raw.strip()]
    if auto_files is None:
        auto_files = auto_assign(crops, [expected_chars(parsed[i]) for i in nonblank],
                                 [plain_first(parsed[i]) for i in nonblank])
    lines, ordinal = [], 0
    for i, raw in enumerate(raw_lines):
        if raw.strip():
            ordinal += 1
            lines.append(new_line(ids, parsed[i], crop=auto_files.get(ordinal)))
        else:
            lines.append(new_line(ids, parsed[i], state="accepted"))
    doc = dict(schema=SCHEMA, tool_version=TOOL_VERSION, slug=SLUG, page=page,
               draft=dict(file=f"{page:04d}.lines.draft.txt", sha256=sha_bytes(draft_bytes), bytes=len(draft_bytes)),
               final_newline=final_nl, revision=0, log_n=0, next_mark=ids.nm, next_line=ids.nl,
               cursor=dict(line=0, mark=None), image=image, crops=crops, lines=lines,
               ignored=[], status="in_progress", created_at=now_iso(), updated_at=now_iso(),
               export=dict(file=f"{page:04d}.lines.txt", sha256=sha_bytes(draft_bytes), prev=None, final=False))
    return doc


def export_text(doc: dict) -> str:
    body = "\n".join(render_segs(l["segs"]) for l in doc["lines"])
    if doc["final_newline"] and doc["lines"]:
        body += "\n"
    return body


def summarize(doc: dict) -> dict:
    markers = sum(unresolved_count(l) for l in doc["lines"])
    lines_left = sum(1 for l in doc["lines"] if l["state"] != "accepted")
    return dict(lines=len(doc["lines"]), markers_left=markers, lines_left=lines_left,
                audited=bool(doc["lines"]) and lines_left == 0 and markers == 0)


def refresh_status(doc: dict) -> None:
    s = summarize(doc)
    doc["status"] = "audited" if s["audited"] else "in_progress"
    doc["export"]["final"] = s["audited"]
    doc["updated_at"] = now_iso()


def public_doc(doc: dict) -> dict:
    d = copy.deepcopy(doc)
    for l in d["lines"]:
        l["out"] = render_segs(l["segs"])
        l["unres"] = unresolved_count(l)
        l["pos"] = mark_positions(l)
        for s in l["segs"]:
            if s["t"] == "mark":
                s["note"] = invisible_note(s["alts"])
    d["pairs"] = {str(k): list(v) for k, v in break_pairs(doc["lines"]).items()}
    d["summary"] = summarize(doc)
    return d


# ---------------------------------------------------------------------------
# Operations. Each returns (lo, n_old, new_lines): replace lines[lo:lo+n_old] by new_lines.
# ---------------------------------------------------------------------------
def _find_mark(line: dict, mid: int) -> dict:
    for s in line["segs"]:
        if s["t"] == "mark" and s["id"] == mid:
            return s
    raise AuditError("that marker is not on this line")


def _reopen(line: dict) -> None:
    line["state"] = "open"


def op_choose(doc, ids, a):
    i = a["line"]
    line = copy.deepcopy(doc["lines"][i])
    seg = _find_mark(line, a["mark"])
    c = a["choice"]
    k = c.get("k")
    if k == "alt":
        if not isinstance(c.get("i"), int) or not 0 <= c["i"] < len(seg["alts"]):
            raise AuditError("no such alternative")
        choice = {"k": "alt", "i": c["i"]}
    elif k == "custom":
        v = c.get("v")
        if not isinstance(v, str) or "\n" in v or "\r" in v or SENT in v:
            raise AuditError("typed text may not contain line breaks")
        choice = {"k": "custom", "v": v}
    elif k in ("unreadable", "empty"):
        choice = {"k": k}
    elif k is None and c.get("clear"):
        choice = None
    else:
        raise AuditError("bad choice")
    seg["choice"] = choice
    _reopen(line)
    return i, 1, [line]


def op_edit(doc, ids, a):
    i = a["line"]
    line = copy.deepcopy(doc["lines"][i])
    si = a["seg"]
    if not (isinstance(si, int) and 0 <= si < len(line["segs"])) or line["segs"][si]["t"] != "text":
        raise AuditError("that is not a plain-text part of the line")
    v = line["segs"][si]["v"]
    st, en, text = a["start"], a["end"], a["text"]
    if not (isinstance(st, int) and isinstance(en, int) and 0 <= st <= en <= len(v)):
        raise AuditError("bad word range")
    if not isinstance(text, str):
        raise AuditError("bad text")
    if text == "":
        if en < len(v) and v[en] == " ":
            en += 1
        elif st > 0 and v[st - 1] == " ":
            st -= 1
    mid = parse_text(text, ids, resolve_unreadable=True, where="typed text: ")
    segs = line["segs"][:si] + [{"t": "text", "v": v[:st]}] + mid + [{"t": "text", "v": v[en:]}] + line["segs"][si + 1:]
    line["segs"] = merge_text(segs)
    _reopen(line)
    return i, 1, [line]


def op_set_text(doc, ids, a):
    i = a["line"]
    line = copy.deepcopy(doc["lines"][i])
    line["segs"] = parse_text(a["text"], ids, resolve_unreadable=True, where="line text: ")
    _reopen(line)
    return i, 1, [line]


def op_split(doc, ids, a):
    i = a["line"]
    line = doc["lines"][i]
    si, off = a["seg"], a.get("offset", 0)
    segs = copy.deepcopy(line["segs"])
    if not (isinstance(si, int) and 0 <= si < len(segs)):
        raise AuditError("bad split position")
    s = segs[si]
    if s["t"] == "text":
        if not (isinstance(off, int) and 0 <= off <= len(s["v"])):
            raise AuditError("bad split position")
        left, right = segs[:si] + [{"t": "text", "v": s["v"][:off]}], [{"t": "text", "v": s["v"][off:]}] + segs[si + 1:]
    else:
        left, right = segs[:si], segs[si:]
    left, right = merge_text(left), merge_text(right)
    # the space where the line is cut disappears with the cut (one space, nothing else)
    if left and left[-1]["t"] == "text" and left[-1]["v"].endswith(" "):
        left[-1] = {"t": "text", "v": left[-1]["v"][:-1]}
    elif right and right[0]["t"] == "text" and right[0]["v"].startswith(" "):
        right[0] = {"t": "text", "v": right[0]["v"][1:]}
    left, right = merge_text(left), merge_text(right)
    if not left or not right:
        raise AuditError("nothing on one side of that position; pick a word in the middle of the line")
    a_line = dict(copy.deepcopy(line), segs=left, state="open")
    b_line = new_line(ids, right, crop=None, crop_src="auto")
    return i, 1, [a_line, b_line]


def op_join(doc, ids, a):
    i = a["line"]
    if i + 1 >= len(doc["lines"]):
        raise AuditError("there is no next line to join with")
    x, y = copy.deepcopy(doc["lines"][i]), copy.deepcopy(doc["lines"][i + 1])
    segs = x["segs"] + ([{"t": "text", "v": " "}] if x["segs"] and y["segs"] else []) + y["segs"]
    x["segs"] = merge_text(segs)
    _reopen(x)
    return i, 2, [x]


def op_insert(doc, ids, a):
    at = a["at"]
    if not (isinstance(at, int) and 0 <= at <= len(doc["lines"])):
        raise AuditError("bad position")
    return at, 0, [new_line(ids, [], crop=None, crop_src="owner")]


def op_delete(doc, ids, a):
    i = a["line"]
    return i, 1, []


def op_accept(doc, ids, a):
    i = a["line"]
    line = copy.deepcopy(doc["lines"][i])
    if unresolved_count(line):
        raise AuditError("this line still has an unresolved marker")
    out = render_segs(line["segs"])
    if (GAP_UNREAD in out or GAP_ABSENT in out) and not a.get("allow_gap_marks"):
        raise AuditError("NEEDS_CONFIRM: this line still contains a … or ∅ left by a reader; replace it, or accept anyway")
    line["state"] = "accepted"
    return i, 1, [line]


def op_reopen(doc, ids, a):
    i = a["line"]
    line = copy.deepcopy(doc["lines"][i])
    line["state"] = "open"
    return i, 1, [line]


def op_break(doc, ids, a):
    i, which = a["line"], a["which"]
    pairs = break_pairs(doc["lines"])
    if i not in pairs:
        raise AuditError("this line has no disputed-break marker")
    first, second = pairs[i]

    def strip(line):
        line = copy.deepcopy(line)
        segs = line["segs"]
        k = next(j for j, s in enumerate(segs) if s["t"] == "brk")
        del segs[k]
        if k < len(segs) and segs[k]["t"] == "text" and segs[k]["v"].startswith(" "):
            segs[k] = {"t": "text", "v": segs[k]["v"][1:]}
        elif k > 0 and segs[k - 1]["t"] == "text" and segs[k - 1]["v"].endswith(" "):
            segs[k - 1] = {"t": "text", "v": segs[k - 1]["v"][:-1]}
        line["segs"] = merge_text(segs)
        _reopen(line)
        return line
    x = strip(doc["lines"][first])
    if second is None:
        return first, 1, [x]
    y = strip(doc["lines"][second])
    if which == "two":
        return first, 2, [x, y]
    if which == "one":
        x["segs"] = merge_text(x["segs"] + ([{"t": "text", "v": " "}] if x["segs"] and y["segs"] else []) + y["segs"])
        return first, 2, [x]
    raise AuditError("choose 'one' or 'two'")


def _shift_range(doc, i, d):
    order = [k["id"] for k in doc["crops"]]
    new = []
    for line in doc["lines"][i:]:
        line = copy.deepcopy(line)
        if line["crop"] in order:
            j = order.index(line["crop"]) + d
            line["crop"] = order[j] if 0 <= j < len(order) else None
            line["crop_src"] = "owner"
        new.append(line)
    return i, len(doc["lines"]) - i, new


def op_set_crop(doc, ids, a):
    """Give line i a crop (or none). With ``following`` the same shift is applied to every later line."""
    i = a["line"]
    c = a.get("crop")
    order = [k["id"] for k in doc["crops"]]
    if c is not None and c not in order:
        raise AuditError("no such crop")
    old = doc["lines"][i]["crop"]
    if a.get("following"):
        if c is None or old not in order:
            raise AuditError("moving the following lines needs a current and a new strip")
        d = order.index(c) - order.index(old)
        if d == 0:
            raise AuditError("that is already this line's strip")
        return _shift_range(doc, i, d)
    line = copy.deepcopy(doc["lines"][i])
    line["crop"], line["crop_src"] = c, "owner"
    return i, 1, [line]


def op_shift_crops(doc, ids, a):
    i, d = a["line"], a["delta"]
    if not isinstance(d, int) or d == 0:
        raise AuditError("bad shift")
    return _shift_range(doc, i, d)


def op_ignore_crop(doc, ids, a):
    c = a.get("crop")
    if c not in {k["id"] for k in doc["crops"]}:
        raise AuditError("no such crop")
    cur = list(doc.get("ignored", []))
    if a.get("ignore", True):
        if c not in cur:
            cur.append(c)
    else:
        cur = [x for x in cur if x != c]
    return 0, 0, [], {"ignored": cur}


def op_add_line_for_crop(doc, ids, a):
    at, c = a.get("at"), a.get("crop")
    if not (isinstance(at, int) and 0 <= at <= len(doc["lines"])):
        raise AuditError("bad position")
    if c not in {k["id"] for k in doc["crops"]}:
        raise AuditError("no such crop")
    cur = [x for x in doc.get("ignored", []) if x != c]
    meta = {"ignored": cur} if cur != list(doc.get("ignored", [])) else None
    res = (at, 0, [new_line(ids, [], crop=c, crop_src="owner")])
    return res + (meta,) if meta is not None else res


OPS = dict(choose=op_choose, edit=op_edit, set_text=op_set_text, split=op_split, join=op_join,
           insert=op_insert, delete=op_delete, accept=op_accept, reopen=op_reopen, brk=op_break,
           set_crop=op_set_crop, shift_crops=op_shift_crops, ignore_crop=op_ignore_crop,
           add_line_for_crop=op_add_line_for_crop)
LINE_OPS_NEED_INDEX = {"choose", "edit", "set_text", "split", "join", "delete", "accept", "reopen", "brk",
                       "set_crop", "shift_crops"}


def replace_region(doc: dict, lo: int, n_old: int, new: list) -> None:
    doc["lines"][lo:lo + n_old] = new


# ---------------------------------------------------------------------------
# Geometry preparation (page image, line finder, crop matching)
# ---------------------------------------------------------------------------
def y_at(xs, ys, x):
    import numpy as np
    x = np.asarray(x, np.float64)
    y = np.interp(x, xs, ys)
    if len(xs) >= 2:
        sl = (ys[1] - ys[0]) / max(1e-9, xs[1] - xs[0])
        sr = (ys[-1] - ys[-2]) / max(1e-9, xs[-1] - xs[-2])
        y = np.where(x < xs[0], ys[0] + sl * (x - xs[0]), y)
        y = np.where(x > xs[-1], ys[-1] + sr * (x - xs[-1]), y)
    return y


def strip_geometry(c: dict) -> dict:
    h = c["up"] + c["dn"]
    up_p, dn_p = c["up"] + CROP_PAD * h, c["dn"] + CROP_PAD * h
    band = up_p + dn_p
    ext = CTX_EXT * band
    margin = 24
    import math
    rows = int(math.ceil(band + 2 * ext)) + 1
    cols = int(math.ceil(c["x1"] - c["x0"] + 2 * margin)) + 1
    return dict(w=cols, h=rows, band_top=round(ext, 1), band_bottom=round(ext + band, 1), margin=margin,
                up_p=round(up_p, 2), dn_p=round(dn_p, 2), ext=round(ext, 2))


def make_strip(gray, c: dict) -> bytes:
    """Straightened strip around crop c with context above and below (PNG bytes)."""
    import fitz
    import numpy as np
    g = c["strip"]
    xs, ys = np.asarray(c["xs"], float), np.asarray(c["ys"], float)
    u = np.arange(g["w"], dtype=np.float64)
    v = np.arange(g["h"], dtype=np.float64)
    X = u[None, :] + c["x0"] - g["margin"] + 0 * v[:, None]
    Y = y_at(xs, ys, X) + v[:, None] - g["up_p"] - g["ext"]
    H, W = gray.shape
    x0, y0 = np.floor(X).astype(int), np.floor(Y).astype(int)
    fx, fy = X - x0, Y - y0
    out = np.zeros_like(X)
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            out += w * np.where(ok, gray[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)], 255.0)
    arr = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    pm = fitz.Pixmap(fitz.csGRAY, arr.shape[1], arr.shape[0], arr.tobytes(), False)
    return pm.tobytes("png")


def load_gray(path: Path):
    import fitz
    import numpy as np
    pm = fitz.Pixmap(str(path))
    if pm.n != 1:
        pm = fitz.Pixmap(fitz.csGRAY, pm)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width).astype(np.float32)


def prepare_geometry(root: Path, page: int, source_root: Path | None = None, log=print):
    """Page image + line-finder geometry + crop matching. Returns (crops, image, auto_files).
    Reads the PDF and runs ``tests/line_finder.py`` (unchanged); writes only <page>.page.png."""
    import fitz
    import numpy as np
    page = check_page(page)
    src = Path(source_root or root)
    ws = src / "books" / SLUG
    P = paths_for(root, page)
    pdf = ws / "source.pdf"
    if not pdf.exists():
        raise AuditError(f"no source PDF at {pdf}")
    doc = fitz.open(str(pdf))
    pg = doc[page - 1]
    render_zoom = 2576 / max(pg.rect.width, pg.rect.height)
    best = None
    for info in pg.get_image_info(xrefs=True):
        bw = info["bbox"][2] - info["bbox"][0]
        if bw <= 0 or not info.get("xref"):
            continue
        ppt = info["width"] / bw
        if best is None or info["width"] * info["height"] > best[1]["width"] * best[1]["height"]:
            best = (ppt, info)
    if best and best[0] > render_zoom:
        if not P["page_png"].exists():
            pix = fitz.Pixmap(doc, best[1]["xref"])
            if pix.alpha or pix.n not in (1, 3):
                pix = fitz.Pixmap(fitz.csRGB, pix)
            P["page_png"].parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(P["page_png"]))
        img_path, kind = P["page_png"], "embedded photo at native resolution"
    else:
        img_path, kind = ws / "pages" / "hires" / f"{page:04d}.png", "workspace hi-res render"
        if not img_path.exists():
            raise AuditError(f"no page image for page {page}")
    pm = fitz.Pixmap(str(img_path))
    gray = load_gray(img_path)
    log(f"page {page}: finding lines on a {pm.width}x{pm.height} image ...")
    import line_finder as LF
    res = LF.find_lines(gray, use_mask=True)
    lines = sorted([L for L in res["lines"] if L.kind == "text"], key=lambda L: float(L.y_at((L.x0 + L.x1) / 2)))
    crops = []
    for k, L in enumerate(lines):
        c = dict(id=f"c{k + 1:02d}", order=k, file=None, kind="X", legacy_line=None,
                 xs=[round(float(x), 1) for x in L.xs], ys=[round(float(y), 1) for y in L.ys],
                 x0=round(float(L.x0), 1), x1=round(float(L.x1), 1), up=round(float(L.up), 2), dn=round(float(L.dn), 2))
        c["strip"] = strip_geometry(c)
        h = L.up + L.dn
        w_, m_, f_ = ink_stats(np.clip(np.rint(copy_line(L, L.up + CROP_PAD * h, L.dn + CROP_PAD * h).rectify(gray)), 0, 255))
        c["ink_w"], c["mass"], c["flat"] = round(w_, 1), int(m_), round(f_, 3)
        crops.append(c)
    # match the crop files made earlier by the draft builder to finder lines (pixel comparison)
    auto = {}
    files = sorted(P["crops"].glob(f"p{page}_[LX]*.png")) if P["crops"].exists() else []
    for f in files:
        m = re.match(rf"p{page}_([LX])(\d+)\.png$", f.name)
        if not m:
            continue
        arr = load_gray(f)
        best_k, best_d = None, 1e9
        for k, L in enumerate(lines):
            h = L.up + L.dn
            up, dn = L.up + CROP_PAD * h, L.dn + CROP_PAD * h
            import math
            shape = (int(math.ceil(up + dn)) + 1, int(math.ceil(L.x1 - L.x0)) + 1)
            if shape != arr.shape:
                continue
            L2 = copy_line(L, up, dn)
            d = float(np.abs(np.clip(np.rint(L2.rectify(gray)), 0, 255) - arr).mean())
            if d < best_d:
                best_k, best_d = k, d
        if best_k is None or best_d > 2.0:
            log(f"  warning: {f.name} did not match any found line (best diff {best_d:.2f})")
            continue
        crops[best_k]["file"] = f"crops/{f.name}"
        crops[best_k]["kind"] = m.group(1)
        if m.group(1) == "L":
            crops[best_k]["legacy_line"] = int(m.group(2))
    image = dict(file=rel_to(img_path, root, src), width=pm.width, height=pm.height, kind=kind, sha256=sha_file(img_path))
    auto = None
    log(f"  {len(crops)} lines found, {sum(1 for c in crops if c['kind'] == 'L')} had draft-builder crop files")
    return crops, image, auto


def ink_stats(arr):
    """(robust ink width in px, ink mass) of a rectified strip: stray marks at the ends are ignored."""
    import numpy as np
    p1, p50 = np.percentile(arr, [2, 50])
    dark = arr < (p50 - 0.5 * (p50 - p1))
    col = dark.sum(0).astype(float)
    sm = np.convolve(col, np.ones(15), "same")
    on = np.where(sm >= 3)[0]
    if not len(on):
        return 1.0, 0.0, 0.0
    clusters, start, prev = [], on[0], on[0]
    for x in on[1:]:
        if x - prev > 90:
            clusters.append((start, prev))
            start = x
        prev = x
    clusters.append((start, prev))
    mass = [float(col[a:b + 1].sum()) for a, b in clusters]
    while len(clusters) > 1:
        tot = sum(mass)
        if mass[0] < 0.04 * tot:
            clusters.pop(0), mass.pop(0)
        elif mass[-1] < 0.04 * tot:
            clusters.pop(), mass.pop()
        else:
            break
    rows = dark.sum(1).astype(float)
    flat = float(np.convolve(rows, np.ones(3), "valid").max() / max(1.0, rows.sum())) if len(rows) >= 3 else 0.0
    return float(clusters[-1][1] - clusters[0][0] + 1), float(sum(mass)), flat


def copy_line(L, up, dn):
    L2 = copy.copy(L)
    L2.up, L2.dn = up, dn
    return L2


def rel_to(p: Path, root: Path, src: Path) -> str:
    p = Path(p).resolve()
    for base in (Path(root).resolve(), Path(src).resolve()):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    raise AuditError(f"image outside the root: {p}")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class Store:
    def __init__(self, root: Path, source_root: Path | None = None, geometry=None, log=print):
        self.root = Path(root)
        self.source_root = Path(source_root or root)
        self.geometry = geometry          # callable(root, page) -> (crops, image, auto); tests inject
        self.log = log
        self.lock = threading.RLock()
        self.docs: dict = {}
        self.grays: dict = {}

    # -- loading / creation -------------------------------------------------
    def _write_state(self, doc, P):
        refresh_status(doc)
        atomic_write(P["state"], canonical(doc))

    def _write_export(self, doc, P, force=False):
        """Rewrite the export atomically. Refuses to overwrite a file that was changed outside the tool."""
        data = export_text(doc).encode("utf-8")
        if P["lines"].exists() and not force:
            cur = sha_file(P["lines"])
            if cur not in (doc["export"]["sha256"], doc["export"].get("prev")):
                raise AuditError(f"{P['lines'].name} was changed outside this tool; refusing to overwrite it. "
                                 f"Move it aside, then restart the tool.")
        new_sha = sha_bytes(data)
        if new_sha != doc["export"]["sha256"]:
            doc["export"]["prev"] = doc["export"]["sha256"]
            doc["export"]["sha256"] = new_sha
        atomic_write(P["lines"], data)

    def _log_records(self, P) -> list:
        if not P["undo"].exists():
            return []
        out = []
        for raw in P["undo"].read_bytes().split(b"\n"):
            if raw.strip():
                out.append(json.loads(raw))
        return out

    def open_page(self, page: int) -> dict:
        page = check_page(page)
        with self.lock:
            if page in self.docs:
                return self.docs[page]
            P = paths_for(self.root, page)
            if P["state"].exists():
                doc = json.loads(P["state"].read_bytes())
                doc.setdefault("ignored", [])
                if not P["draft"].exists():
                    raise AuditError(f"the draft copy {P['draft'].name} is missing; refusing to continue")
                if sha_file(P["draft"]) != doc["draft"]["sha256"]:
                    raise AuditError(f"{P['draft'].name} does not match the draft this state was created from; refusing")
                self._heal(doc, P)
            else:
                if P["draft"].exists():
                    src = P["draft"].read_bytes()
                    if P["lines"].exists() and P["lines"].read_bytes() != src:
                        raise AuditError(f"{P['lines'].name} differs from {P['draft'].name} but no state file exists; "
                                         f"refusing to overwrite it. Move one of them aside.")
                else:
                    if not P["lines"].exists():
                        raise AuditError(f"no draft at {P['lines']}")
                    src = P["lines"].read_bytes()
                try:
                    if self.geometry is not None:
                        crops, image, auto = self.geometry(self.root, page)
                    else:
                        crops, image, auto = prepare_geometry(self.root, page, self.source_root, self.log)
                except AuditError:
                    raise
                doc = build_state(page, src, crops, image, auto)
                if not P["draft"].exists():
                    with P["draft"].open("xb") as f:
                        f.write(src)
                        f.flush()
                        os.fsync(f.fileno())
                    if sha_file(P["draft"]) != doc["draft"]["sha256"]:
                        raise AuditError("the draft copy did not verify after writing")
                self._write_state(doc, P)
                self._write_export(doc, P)
            self.docs[page] = doc
            return doc

    def _heal(self, doc, P):
        recs = self._log_records(P)
        if len(recs) < doc["log_n"]:
            doc["log_n"] = len(recs)        # log lost its last record: those actions cannot be undone
        changed = False
        for r in recs[doc["log_n"]:]:
            if doc["lines"][r["lo"]:r["lo"] + len(r["old"])] != r["old"]:
                raise AuditError("undo log and state disagree; refusing to continue")
            replace_region(doc, r["lo"], len(r["old"]), r["new"])
            if r.get("meta_new") is not None:
                doc["ignored"] = list(r["meta_new"]["ignored"])
            doc["revision"] += 1
            doc["log_n"] = r["seq"]
            doc["next_mark"], doc["next_line"] = r["ids"]
            changed = True
        if changed:
            self._write_state(doc, P)
        # a crash between state and export leaves the export one step behind
        want = sha_bytes(export_text(doc).encode("utf-8"))
        if not P["lines"].exists() or sha_file(P["lines"]) != want:
            if not P["lines"].exists() or sha_file(P["lines"]) in (doc["export"]["sha256"], doc["export"].get("prev")):
                self._write_export(doc, P)
                self._write_state(doc, P)
            else:
                raise AuditError(f"{P['lines'].name} was changed outside this tool; refusing to overwrite it.")

    # -- actions ------------------------------------------------------------
    def get(self, page: int) -> dict:
        with self.lock:
            return public_doc(self.open_page(page))

    def act(self, page: int, op: str, args: dict, rev) -> dict:
        page = check_page(page)
        with self.lock:
            doc = self.open_page(page)
            P = paths_for(self.root, page)
            if op == "cursor":
                cur = dict(line=int(args.get("line", 0)), mark=args.get("mark"))
                doc["cursor"] = cur
                self._write_state(doc, P)
                return public_doc(doc)
            if op not in OPS:
                raise AuditError(f"unknown action {op!r}")
            if rev is None or int(rev) != doc["revision"]:
                raise Conflict(public_doc(doc))
            args = dict(args or {})
            n = len(doc["lines"])
            if op in LINE_OPS_NEED_INDEX:
                i = args.get("line")
                if not (isinstance(i, int) and 0 <= i < n):
                    raise AuditError("no such line")
                if "lid" in args and doc["lines"][i]["id"] != args["lid"]:
                    raise Conflict(public_doc(doc))
            ids = Ids(doc["next_mark"], doc["next_line"])
            res = OPS[op](doc, ids, args)
            lo, n_old, new = res[:3]
            old = copy.deepcopy(doc["lines"][lo:lo + n_old])
            new = copy.deepcopy(new)
            rec = dict(seq=doc["log_n"] + 1, kind="op", at=now_iso(), op=op, args=args, lo=lo, old=old, new=new,
                       ids=[ids.nm, ids.nl], rev_after=doc["revision"] + 1)
            if len(res) > 3 and res[3] is not None:
                rec["meta_old"] = dict(ignored=list(doc.get("ignored", [])))
                rec["meta_new"] = res[3]
            self._commit(doc, P, rec, ids)
            doc["cursor"] = dict(line=min(max(lo, 0), max(0, len(doc["lines"]) - 1)), mark=None) if op in (
                "split", "join", "insert", "delete", "brk") else doc["cursor"]
            self._write_state(doc, P)
            return public_doc(doc)

    def _commit(self, doc, P, rec, ids=None):
        """log first, then state, then export (each atomic/fsynced)."""
        append_jsonl(P["undo"], rec)
        replace_region(doc, rec["lo"], len(rec["old"]), copy.deepcopy(rec["new"]))
        if rec.get("meta_new") is not None:
            doc["ignored"] = list(rec["meta_new"]["ignored"])
        doc["revision"] += 1
        doc["log_n"] = rec["seq"]
        doc["next_mark"], doc["next_line"] = rec["ids"]
        self._write_state(doc, P)
        self._write_export(doc, P)
        self._write_state(doc, P)        # records the export hashes

    def active_stack(self, P) -> list:
        stack = []
        for r in self._log_records(P):
            if r["kind"] == "op":
                stack.append(r)
            elif stack and stack[-1]["seq"] == r["of"]:
                stack.pop()
        return stack

    def undo(self, page: int, rev) -> tuple:
        page = check_page(page)
        with self.lock:
            doc = self.open_page(page)
            P = paths_for(self.root, page)
            if rev is None or int(rev) != doc["revision"]:
                raise Conflict(public_doc(doc))
            stack = self.active_stack(P)
            if not stack:
                raise AuditError("nothing to undo")
            top = stack[-1]
            if doc["lines"][top["lo"]:top["lo"] + len(top["new"])] != top["new"]:
                raise AuditError("the page no longer matches the last logged action; cannot undo safely")
            rec = dict(seq=doc["log_n"] + 1, kind="undo", of=top["seq"], at=now_iso(), op="undo:" + top["op"],
                       lo=top["lo"], old=top["new"], new=top["old"], ids=[doc["next_mark"], doc["next_line"]],
                       rev_after=doc["revision"] + 1)
            if top.get("meta_new") is not None:
                if list(doc.get("ignored", [])) != list(top["meta_new"]["ignored"]):
                    raise AuditError("the page no longer matches the last logged action; cannot undo safely")
                rec["meta_old"], rec["meta_new"] = top["meta_new"], top["meta_old"]
            self._commit(doc, P, rec)
            doc["cursor"] = dict(line=min(top["lo"], max(0, len(doc["lines"]) - 1)), mark=None)
            self._write_state(doc, P)
            return public_doc(doc), top["op"]

    def can_undo(self, page: int) -> int:
        with self.lock:
            self.open_page(page)
            return len(self.active_stack(paths_for(self.root, page)))

    def pages(self) -> list:
        out = []
        for p in ALLOWED_PAGES:
            try:
                d = self.open_page(p)
                s = summarize(d)
                out.append(dict(page=p, label=f"Page {p}", lines=s["lines"], markers_left=s["markers_left"],
                                lines_left=s["lines_left"], status=d["status"]))
            except AuditError as exc:
                out.append(dict(page=p, label=f"Page {p}", error=str(exc)))
        return out

    def gray(self, page: int):
        with self.lock:
            if page not in self.grays:
                doc = self.open_page(page)
                self.grays[page] = load_gray(self.image_path(doc))
            return self.grays[page]

    def image_path(self, doc) -> Path:
        for base in (self.root, self.source_root):
            p = (base / doc["image"]["file"]).resolve()
            if p.exists():
                return p
        raise AuditError("page image is missing")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def make_handler(store: Store, port: int):
    from text_audit_ui import PAGE_HTML

    class Handler(BaseHTTPRequestHandler):
        server_version = "text-audit/1"

        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, v):
            self._send(code, json.dumps(v, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _page(self, q):
            try:
                return check_page(int(q.get("page", [""])[0]))
            except ValueError:
                raise AuditError("bad page")

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    html = PAGE_HTML.replace("__PORT__", str(port)).replace("__CMD__", restart_command(store.root, port))
                    return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                if u.path == "/api/ping":
                    return self._json(200, dict(ok=True, tool=TOOL_VERSION))
                if u.path == "/api/pages":
                    return self._json(200, dict(ok=True, pages=store.pages()))
                if u.path == "/api/page":
                    p = self._page(q)
                    return self._json(200, dict(ok=True, doc=store.get(p), undo_depth=store.can_undo(p)))
                if u.path == "/img":
                    p = self._page(q)
                    return self._send(200, store.image_path(store.open_page(p)).read_bytes(), "image/png", cache=True)
                if u.path == "/crop":
                    p = self._page(q)
                    doc = store.open_page(p)
                    name = q.get("file", [""])[0]
                    if name not in {c["file"] for c in doc["crops"] if c["file"]}:
                        raise AuditError("no such crop")
                    f = (paths_for(store.root, p)["crops"].parent / name).resolve()
                    return self._send(200, f.read_bytes(), "image/png", cache=True)
                if u.path == "/strip":
                    p = self._page(q)
                    doc = store.open_page(p)
                    cid = q.get("crop", [""])[0]
                    c = next((c for c in doc["crops"] if c["id"] == cid), None)
                    if c is None:
                        raise AuditError("no such crop")
                    return self._send(200, make_strip(store.gray(p), c), "image/png", cache=True)
                if u.path == "/font.ttf" and FONT.exists():
                    return self._send(200, FONT.read_bytes(), "font/ttf", cache=True)
                return self._json(404, dict(ok=False, error="not found"))
            except AuditError as exc:
                return self._json(400, dict(ok=False, error=str(exc)))
            except Exception as exc:
                return self._json(500, dict(ok=False, error=f"{type(exc).__name__}: {exc}"))

        def do_POST(self):
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                page = check_page(body.get("page"))
                if u.path == "/api/op":
                    doc = store.act(page, str(body.get("op")), body.get("args") or {}, body.get("rev"))
                    return self._json(200, dict(ok=True, doc=doc, undo_depth=store.can_undo(page)))
                if u.path == "/api/undo":
                    doc, op = store.undo(page, body.get("rev"))
                    return self._json(200, dict(ok=True, doc=doc, undone=op, undo_depth=store.can_undo(page)))
                return self._json(404, dict(ok=False, error="not found"))
            except Conflict as c:
                return self._json(409, dict(ok=False, error=str(c), doc=c.doc))
            except AuditError as exc:
                return self._json(400, dict(ok=False, error=str(exc)))
            except Exception as exc:
                return self._json(500, dict(ok=False, error=f"{type(exc).__name__}: {exc}"))

    return Handler


def restart_command(root: Path, port: int) -> str:
    cmd = f"cd {REPO} && ./venv/bin/python tests/text_audit.py serve --port {port}"
    if Path(root).resolve() != REPO:
        cmd += f" --root {Path(root).resolve()}"
    return cmd


def serve(root: Path, port: int, open_browser: bool, source_root: Path | None = None) -> None:
    store = Store(root, source_root)
    print(f"Preparing pages (the first start finds the printed lines on each page; this takes a minute or two) ...",
          flush=True)
    for p in store.pages():
        if "error" in p:
            print(f"  {p['label']}: ERROR {p['error']}")
        else:
            print(f"  {p['label']}: {p['lines']} lines, {p['markers_left']} markers left, "
                  f"{p['lines_left']} lines not accepted, {p['status']}")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, port))
    url = f"http://127.0.0.1:{port}/"
    print(f"Text audit: {url}\nEvery action is saved at once. Ctrl-C stops the server.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "status", "prepare"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=str(REPO))
        s.add_argument("--source-root", default=None, help="where source.pdf and pages/ are read from (default: --root)")
        if name == "serve":
            s.add_argument("--port", type=int, default=DEFAULT_PORT)
            s.add_argument("--open", action="store_true")
        if name == "prepare":
            s.add_argument("--page", type=int, required=True)
    a = ap.parse_args(argv)
    root = Path(a.root)
    src = Path(a.source_root) if a.source_root else None
    try:
        if a.cmd == "serve":
            serve(root, a.port, a.open, src)
        elif a.cmd == "status":
            for p in Store(root, src).pages():
                print(p)
        elif a.cmd == "prepare":
            Store(root, src).open_page(a.page)
            print("prepared")
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
