"""Step 2 (paid): does Gemini/Opus word disagreement keep the owner's finds and drop the false alarms?

  ./venv/bin/python tests/review_disagreement_step2.py dry     # no network: inputs, decisions, worst-case cost
  ./venv/bin/python tests/review_disagreement_step2.py read    # one Opus 5.5 band reading per page (paid, cached)
  ./venv/bin/python tests/review_disagreement_step2.py score   # $0, from the cache

Book: books/bachehaye_ghali-gemini/ (read only), the owner's 20 reviewed pages.
Primary = Gemini's first-pass Markdown as it was before review (text/NNNN.orig.md when it
exists, else text/NNNN.md). Second = farsi2epub.hard_reader.read_page(..., reader="opus")
exactly (production, arm t1r_opus), cached under out/review_step2/readings/pNNNN.json; a
cached page is never requested again. Ledger out/review_step2/ledger.jsonl; hard cap
$2.00 of Anthropic spend, checked before every call with a worst-case reservation
(max_tokens output and an upper bound on input tokens, times the retry attempts that
can be billed); a failed or uncertain call is counted at its full reservation. No Google
call is ever made (Gemini keys are blanked in this process).
Flags = TE.t2_flags(TE.tokenize_md(primary), TE.tokenize_lines(opus_lines)), as
tests/review_disagreement_probe.py. Flag indices are in t2_flags' ortho-folded token space,
so correction spans are mapped into that same space (prefix tokens, folded).
The owner's decisions: the review corrections are rebuilt from the pre-review state on a
temporary copy of the workspace (review._make_review_refiner offline + feedback.page_snapshot),
APPROVED = a text_edit record in box_truth.jsonl (verdict approved/edited) for its
correction_key, plus the page-87 correction whose old text is «باز» (owner: rejected by
mistake). Prints counts only, never page text.

CRITERIA, FIXED 2026-11-17 BEFORE THE FIRST READING (owner rule D25). One Opus band reading per page (budget allows one run; stated as a limitation). Population: word-level corrections the old reviewer suggested on the 20 pages; truth = the owner's decisions.
  KEEP   approved word-level corrections that lie at a flag. PASS: at most one approved word-level correction missed (and at least 2 exist). FAIL otherwise.
  DROP   rejected word-level corrections that lie at no flag. PASS: >= 80%. (The old reviewer: 0% by construction, every one was suggested.)
  BURDEN flags / primary words, pooled over the 20 pages. PASS: <= 5%.
  NEW    flags away from every correction are counted and listed per page, not judged (the owner has not looked there).
  Outcome: all three PASS -> propose replacing the old reviewer's suggestions in review with these flags (both readings shown, no suggested fix), plus a separate structure check for dividers. Otherwise report which part failed; nothing is tuned on these pages.

ADDENDUM, 2026-11-17, STILL BEFORE ANY READING (no Opus reading exists yet; the criteria above stand
and decide the outcome). The dry run showed that 2 of the 6 approved word-level corrections change
only joins, ZWNJ or punctuation, which the lenient flags ignore by design, and that 31 of the 53
rejected word-level corrections are findings with no suggested text. So two figures are printed
beside the verdict, as information only, not criteria:
  KEEP-W  approved corrections that change a word (lenient-visible), at a flag.
  DROP-R  rejected corrections that suggested replacement text, at no flag.
The approved join/ZWNJ/punctuation-only corrections are a known blind spot of this method and are
reported as such.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
    os.environ[_k] = ""                       # never call Google (load_env does not override set keys)
CMD = sys.argv[1] if len(sys.argv) > 1 else ""
if CMD != "read":
    os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"   # no network outside `read`

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import yaml  # noqa: E402

import transcription_experiments as TE  # noqa: E402
from farsi2epub import config, hard_reader  # noqa: E402

SLUG = "bachehaye_ghali-gemini"
BOOK = ROOT / "books" / SLUG
PAGES = [12, 17, 23, 29, 41, 44, 46, 49, 56, 61, 70, 72, 74, 77, 79, 80, 82, 86, 87, 88]
OUT = ROOT / "out" / "review_step2"
READ_DIR = OUT / "readings"
LEDGER = OUT / "ledger.jsonl"
CORR_FILE = OUT / "corrections.json"         # rebuilt corrections (contains page text; stays under out/)
RESULT = OUT / "result.json"
CAP_USD = 2.00
RETRY_ATTEMPTS = 3                            # llm._call_with_retry default: validation failures are billed
SAFETY = 1.10
P87_OLD = "باز"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def primary_md(n: int) -> tuple[str, str]:
    o = BOOK / "text" / f"{n:04d}.orig.md"
    if o.is_file():
        return o.read_text(encoding="utf-8"), "orig.md"
    return (BOOK / "text" / f"{n:04d}.md").read_text(encoding="utf-8"), "md"


def book_meta() -> dict:
    return yaml.safe_load((BOOK / "book.yaml").read_text(encoding="utf-8")) or {}


def hires(n: int) -> bytes:
    return (BOOK / "pages" / "hires" / f"{n:04d}.png").read_bytes()


def worst_case(n: int, meta: dict) -> tuple[float, dict]:
    """Upper bound on the cost of one read_page call (all billable attempts)."""
    req = hard_reader.build_request(BOOK / "source.pdf", n, int(meta["page_count"]), meta.get("title_fa"),
                                    hires(n), reader="opus")
    img_tok = sum(config.image_tokens(c.width, c.height) for c in req["content"]
                  if isinstance(c, hard_reader.BandImage))
    text_bytes = len(req["system"].encode()) + sum(len(c.encode()) for c in req["content"] if isinstance(c, str))
    schema_bytes = len(json.dumps(req["schema"].model_json_schema()).encode())
    tok_in = img_tok + text_bytes + schema_bytes + 2000       # 1 token per byte is a safe bound; + framing
    pr = config.PRICES[req["model"]]
    one = (tok_in / 1e6 * pr["in"] + req["max_tokens"] / 1e6 * pr["out"]) * SAFETY
    return one * RETRY_ATTEMPTS, {"model": req["model"], "bands": req["info"]["bands"], "source": req["info"]["source"],
                                  "image_tokens": img_tok, "input_bound": tok_in, "max_tokens": req["max_tokens"],
                                  "one_attempt": one}


# ---------------------------------------------------------------------------
# corrections and the owner's decisions
# ---------------------------------------------------------------------------

def rebuild_corrections() -> dict:
    """Review corrections from the pre-review state, on a temporary workspace copy."""
    tmp = OUT / "ws" / "books"
    dst = tmp / SLUG
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "locate_evidence").mkdir(parents=True)
    (dst / "pages" / "hires").mkdir(parents=True)
    shutil.copytree(BOOK / "text", dst / "text")
    gb = BOOK / "locate_evidence" / "gemini_boxes"
    if gb.is_dir():
        shutil.copytree(gb, dst / "locate_evidence" / "gemini_boxes")
    shutil.copy(BOOK / "book.yaml", dst / "book.yaml")
    shutil.copy(BOOK / "source.pdf", dst / "source.pdf")
    for n in PAGES:
        for rel in (f"pages/{n:04d}.png", f"pages/hires/{n:04d}.png"):
            if (BOOK / rel).is_file():
                shutil.copy(BOOK / rel, dst / rel)
        o = dst / "text" / f"{n:04d}.orig.md"
        if o.exists():
            (dst / "text" / f"{n:04d}.md").write_text(o.read_text(encoding="utf-8"), encoding="utf-8")
        p = dst / "text" / f"{n:04d}.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        s.setdefault("qc", {})["suggestion_status"] = "pending"
        p.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    from farsi2epub import feedback, review
    from farsi2epub.workspace import Workspace
    ws = Workspace(SLUG, tmp)
    review._REFINER = review._make_review_refiner(ws, config.MODEL_STRONG, "offline", 0.0, engine="gemini")
    out = {}
    for n in PAGES:
        snap = feedback.page_snapshot(ws, n)
        md_pre, _ = primary_md(n)
        assert snap is not None and snap["text"] == md_pre, f"page {n}: snapshot text is not the pre-review text"
        out[str(n)] = [{"view_key": vk, "correction_key": c["correction_key"],
                        "target": {k: c["target"].get(k) for k in ("kind", "old", "suggested", "span", "issue_type")}}
                       for vk, c in snap["corrections"].items()]
    CORR_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def has_letter(s: str) -> bool:
    for ch in s or "":
        cp = ord(ch)
        if unicodedata.category(ch).startswith("L") and (
                0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF
                or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF):
            return True
    return False


def decisions(corr: dict) -> tuple[dict, dict]:
    rows = [json.loads(l) for l in (ROOT / "box_truth.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    te = [r for r in rows if r.get("book") == SLUG and r.get("family") == "text_edit"
          and (r.get("human") or {}).get("verdict") in ("approved", "edited")]
    keys_all = {c["correction_key"]: (int(p), c) for p, cs in corr.items() for c in cs}
    approved = {r["correction_key"] for r in te if r["correction_key"] in keys_all}
    unmatched = [{"page": r.get("page"), "key": r["correction_key"]} for r in te if r["correction_key"] not in keys_all]
    p87 = [c for c in corr.get("87", []) if (c["target"]["old"] or "").strip() == P87_OLD]
    for c in p87:
        approved.add(c["correction_key"])
    info = {"text_edit_records": len(te), "text_edit_keys": len({r["correction_key"] for r in te}),
            "text_edit_matched": len({r["correction_key"] for r in te} & set(keys_all)),
            "text_edit_unmatched": unmatched, "p87_baz_found": len(p87),
            "p87_baz_already_text_edit": sum(1 for c in p87 if c["correction_key"] in
                                             {r["correction_key"] for r in te})}
    return {k: (k in approved) for k in keys_all}, info


def classify(corr: dict) -> list[dict]:
    appr, _ = decisions(corr)
    out = []
    for p, cs in corr.items():
        for c in cs:
            t = c["target"]
            word = has_letter(t.get("old") or "") or has_letter(t.get("suggested") or "")
            out.append({"page": int(p), "key": c["correction_key"], "approved": appr[c["correction_key"]],
                        "level": "word" if word else "structure", "kind": t.get("kind"),
                        "issue_type": t.get("issue_type"), "has_span": t.get("span") is not None})
    return out


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def ledger_rows() -> list[dict]:
    if not LEDGER.is_file():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def committed() -> tuple[float, float]:
    """(spend counted against the cap, of which actual settled). An open or failed
    reservation counts in full."""
    res, settled = {}, {}
    for r in ledger_rows():
        if r["event"] == "reserve":
            res[r["call_id"]] = r["reserved"]
        elif r["event"] == "settle":
            settled[r["call_id"]] = r
    total = actual = 0.0
    for cid, amt in res.items():
        s = settled.get(cid)
        if s is None or s["status"] != "ok":
            total += amt
        else:
            total += s["cost"]
            actual += s["cost"]
    return total, actual


def log(rec: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_dry() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = book_meta()
    print(f"book {SLUG}: page_count {meta.get('page_count')}, title present: {bool(meta.get('title_fa'))}")
    total, total_one, todo = 0.0, 0.0, 0
    print(f"{'page':>4s} {'primary':8s} {'words':>5s} {'hires':>5s} {'cached':>6s} {'bands':>5s} {'source':15s} "
          f"{'img_tok':>7s} {'in_bound':>8s} {'1 attempt $':>11s} {'worst $':>8s}")
    for n in PAGES:
        md, src = primary_md(n)
        words = len(TE.fold_tokens(TE.tokenize_md(md), "ortho"))
        cached = (READ_DIR / f"p{n:04d}.json").is_file()
        wc, d = worst_case(n, meta)
        if not cached:
            total += wc
            total_one += d["one_attempt"]
            todo += 1
        print(f"{n:4d} {src:8s} {words:5d} {'yes' if (BOOK / 'pages/hires' / f'{n:04d}.png').is_file() else 'NO':>5s} "
              f"{'yes' if cached else 'no':>6s} {d['bands']:5d} {d['source']:15s} {d['image_tokens']:7d} "
              f"{d['input_bound']:8d} {d['one_attempt']:11.4f} {wc:8.4f}")
    corr = rebuild_corrections()
    rows = classify(corr)
    _, info = decisions(corr)
    print(f"\ncorrections rebuilt: {len(rows)} on {sum(1 for p in corr if corr[p])} pages "
          f"(per page: " + ", ".join(f"{p}:{len(corr[p])}" for p in corr) + ")")
    w = [r for r in rows if r["level"] == "word"]
    s = [r for r in rows if r["level"] == "structure"]
    print(f"  word-level {len(w)}: approved {sum(r['approved'] for r in w)}, rejected {sum(not r['approved'] for r in w)}")
    print(f"  structure  {len(s)}: approved {sum(r['approved'] for r in s)}, rejected {sum(not r['approved'] for r in s)}")
    print(f"  without char span (old text located later or 'no span'): {sum(not r['has_span'] for r in rows)}")
    print(f"  text_edit records (approved/edited) {info['text_edit_records']}, distinct keys {info['text_edit_keys']}, "
          f"matched {info['text_edit_matched']}, unmatched {len(info['text_edit_unmatched'])}")
    print(f"  page-87 '{'old = baz'}' corrections found: {info['p87_baz_found']} "
          f"(already a text_edit: {info['p87_baz_already_text_edit']})")
    done_total, done_actual = committed()
    print(f"\npages to read {todo}; worst case one attempt each ${total_one:.2f}; worst case with "
          f"{RETRY_ATTEMPTS} billable attempts ${total:.2f}; ledger so far ${done_total:.4f} counted "
          f"(${done_actual:.4f} settled); cap ${CAP_USD:.2f}")
    if done_total + total > CAP_USD:
        print(f"REFUSE: total worst case ${done_total + total:.2f} > cap ${CAP_USD:.2f}; `read` will not run "
              f"without --sequential (per-call reservation against the cap, may stop before all pages)")
        return 2
    print("OK: total worst case within the cap")
    return 0


def _is_credit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    code = getattr(exc, "status_code", None)
    return code in (401, 403) or any(w in msg for w in ("credit", "billing", "balance", "permission", "access"))


def cmd_read(sequential: bool) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    READ_DIR.mkdir(parents=True, exist_ok=True)
    meta = book_meta()
    todo = [n for n in PAGES if not (READ_DIR / f"p{n:04d}.json").is_file()]
    worst = {n: worst_case(n, meta)[0] for n in todo}
    done_total, _ = committed()
    if done_total + sum(worst.values()) > CAP_USD and not sequential:
        print(f"REFUSE: total worst case ${done_total + sum(worst.values()):.2f} > cap ${CAP_USD:.2f} (pass --sequential "
              "to read page by page under a per-call reservation)")
        return 2
    from farsi2epub import llm
    llm.load_env()
    client = llm.get_client()
    for n in todo:
        res = worst[n]
        done_total, done_actual = committed()
        if done_total + res > CAP_USD:
            print(f"STOP before page {n}: counted ${done_total:.4f} + reservation ${res:.4f} > cap ${CAP_USD:.2f}")
            return 3
        cid = f"p{n:04d}-{int(time.time() * 1000)}"
        log({"event": "reserve", "call_id": cid, "page": n, "reserved": round(res, 6)})
        try:
            lines, info, usage, cost = hard_reader.read_page(client, BOOK / "source.pdf", n, int(meta["page_count"]),
                                                              meta.get("title_fa"), hires(n), reader="opus")
        except Exception as exc:  # noqa: BLE001
            log({"event": "settle", "call_id": cid, "page": n, "status": "failed", "cost": round(res, 6),
                 "error": type(exc).__name__, "http_status": getattr(exc, "status_code", None)})
            print(f"page {n}: FAILED ({type(exc).__name__}, status {getattr(exc, 'status_code', None)}); "
                  f"counted at its reservation ${res:.4f}")
            if _is_credit_error(exc):
                print("credit/billing/access error: stopping")
                return 4
            continue
        rec = {"page": n, "model": config.MODEL_HARD, "reader_version": hard_reader.READER_VERSION,
               "lines": lines, "info": info, "usage": usage, "cost": cost,
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        tmpf = READ_DIR / f"p{n:04d}.json.tmp"
        tmpf.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmpf.replace(READ_DIR / f"p{n:04d}.json")
        log({"event": "settle", "call_id": cid, "page": n, "status": "ok", "cost": round(cost, 6), "usage": usage})
        t, a = committed()
        print(f"page {n}: ok, {len(lines)} lines, in {usage['input_tokens']} out {usage['output_tokens']} tok, "
              f"${cost:.4f}; counted ${t:.4f}")
    t, a = committed()
    print(f"read done; counted ${t:.4f}, settled ${a:.4f}")
    return 0


def _folded_count(md_prefix: str) -> int:
    return len(TE.fold_tokens(TE.tokenize_md(md_prefix), "ortho"))


def span_tokens(md: str, span) -> tuple[set, set]:
    a, b = int(span[0]), int(span[1])
    start = _folded_count(md[:a])
    if 0 < a < len(md) and not md[a - 1].isspace() and not md[a].isspace():
        start = max(0, start - 1)             # span starts inside a word: the prefix counted its first half
    end = _folded_count(md[:b])
    if end <= start:
        toks = {start - 1, start} - {-1}
        gaps = {start}
    else:
        toks = set(range(start, end))
        gaps = set(range(start, end + 1))
    return toks, gaps


def cmd_score() -> int:
    corr = rebuild_corrections()
    rows = classify(corr)
    _, info = decisions(corr)
    by_key = {c["correction_key"]: c for cs in corr.values() for c in cs}
    pages, missing = [], []
    for n in PAGES:
        f = READ_DIR / f"p{n:04d}.json"
        md, _ = primary_md(n)
        prim = TE.tokenize_md(md)
        nwords = len(TE.fold_tokens(prim, "ortho"))
        prow = {"page": n, "primary_words": nwords, "read": f.is_file()}
        if not f.is_file():
            missing.append(n)
            pages.append(prow)
            for r in rows:
                if r["page"] == n:
                    r["at_flag"] = None
                    r["span_source"] = "no reading"
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        ftok, fgap = TE.t2_flags(prim, TE.tokenize_lines(rec["lines"]))
        cov_t, cov_g = set(), set()
        for r in [r for r in rows if r["page"] == n]:
            t = by_key[r["key"]]["target"]
            span, src = t.get("span"), "span"
            if span is None:
                old = t.get("old") or ""
                if old.strip() and md.count(old) == 1:
                    i = md.index(old)
                    span, src = [i, i + len(old)], "old located"
                else:
                    src = "no span"
            r["span_source"] = src
            if span is None:
                r["at_flag"] = None
                continue
            toks, gaps = span_tokens(md, span)
            cov_t |= toks
            cov_g |= gaps
            r["at_flag"] = bool(toks & ftok or gaps & fgap)
        new_t, new_g = ftok - cov_t, fgap - cov_g
        prow.update(flags=len(ftok) + len(fgap), flagged_words=len(ftok), flagged_gaps=len(fgap),
                    new_flags=len(new_t) + len(new_g), opus_lines=len(rec["lines"]), cost=rec.get("cost"))
        pages.append(prow)

    print(f"{'page':>4s} {'words':>5s} {'flags':>5s} {'(w/g)':>7s} {'burden':>7s} {'corr':>4s} {'appr':>4s} "
          f"{'w-appr@flag':>11s} {'w-rej@noflag':>12s} {'struct':>6s} {'nospan':>6s} {'NEW':>4s}")
    for p in pages:
        n = p["page"]
        pr = [r for r in rows if r["page"] == n]
        wa = [r for r in pr if r["level"] == "word" and r["approved"] and r["at_flag"] is not None]
        wr = [r for r in pr if r["level"] == "word" and not r["approved"] and r["at_flag"] is not None]
        if not p["read"]:
            print(f"{n:4d} {p['primary_words']:5d}  NO READING")
            continue
        print(f"{n:4d} {p['primary_words']:5d} {p['flags']:5d} {p['flagged_words']:3d}/{p['flagged_gaps']:<3d} "
              f"{100 * p['flags'] / max(1, p['primary_words']):6.1f}% {len(pr):4d} {sum(r['approved'] for r in pr):4d} "
              f"{sum(r['at_flag'] for r in wa):5d}/{len(wa):<5d} {sum(not r['at_flag'] for r in wr):6d}/{len(wr):<5d} "
              f"{sum(r['level'] == 'structure' for r in pr):6d} {sum(r['span_source'] == 'no span' for r in pr):6d} "
              f"{p['new_flags']:4d}")

    judged = [r for r in rows if r["at_flag"] is not None]
    wa = [r for r in judged if r["level"] == "word" and r["approved"]]
    wr = [r for r in judged if r["level"] == "word" and not r["approved"]]
    st = [r for r in rows if r["level"] == "structure"]
    nospan = [r for r in rows if r.get("span_source") == "no span"]
    read_pages = [p for p in pages if p["read"]]
    flags = sum(p["flags"] for p in read_pages)
    words = sum(p["primary_words"] for p in read_pages)
    keep_hit = sum(r["at_flag"] for r in wa)
    drop_hit = sum(not r["at_flag"] for r in wr)
    print(f"\npages read {len(read_pages)}/{len(PAGES)}" + (f"; missing {missing}" if missing else ""))
    print(f"corrections {len(rows)}: word-level {sum(r['level'] == 'word' for r in rows)}, structure {len(st)} "
          f"(approved {sum(r['approved'] for r in st)}, at a flag {sum(bool(r['at_flag']) for r in st)}), "
          f"no span {len(nospan)} (approved {sum(r['approved'] for r in nospan)}), "
          f"old text located {sum(r.get('span_source') == 'old located' for r in rows)}")
    print(f"KEEP   approved word-level at a flag: {keep_hit}/{len(wa)} (missed {len(wa) - keep_hit})")
    print(f"DROP   rejected word-level at no flag: {drop_hit}/{len(wr)}"
          + (f" = {100 * drop_hit / len(wr):.1f}%" if wr else ""))
    def _fold(x):
        return [t.text for t in TE.fold_tokens(TE.tokenize_md(x or ""), "ortho")]

    def _word_change(r):
        t = by_key[r["key"]]["target"]
        return t.get("suggested") is not None and _fold(t.get("old")) != _fold(t.get("suggested"))
    kw = [r for r in wa if _word_change(r)]
    dr = [r for r in wr if by_key[r["key"]]["target"].get("suggested") is not None]
    print(f"KEEP-W (information) approved corrections that change a word, at a flag: "
          f"{sum(r['at_flag'] for r in kw)}/{len(kw)}; approved join/ZWNJ/punctuation-only: "
          f"{len(wa) - len(kw)} (at a flag {sum(r['at_flag'] for r in wa if r not in kw)}), a blind spot by design")
    print(f"DROP-R (information) rejected corrections with suggested text, at no flag: "
          f"{sum(not r['at_flag'] for r in dr)}/{len(dr)}; findings without suggested text: "
          f"{sum(not r['at_flag'] for r in wr if r not in dr)}/{len(wr) - len(dr)} at no flag")
    print(f"BURDEN flags / primary words: {flags}/{words}" + (f" = {100 * flags / words:.2f}%" if words else ""))
    print(f"NEW    flags away from every correction: {sum(p.get('new_flags', 0) for p in read_pages)} "
          f"(per page: " + ", ".join(f"{p['page']}:{p['new_flags']}" for p in read_pages) + ")")
    print(f"decisions: text_edit records {info['text_edit_records']}, matched {info['text_edit_matched']}, "
          f"unmatched {len(info['text_edit_unmatched'])}; page-87 baz corrections {info['p87_baz_found']}")

    if missing:
        verdict = {"outcome": f"NOT JUDGED: pages without a reading {missing}"}
        print("\n" + verdict["outcome"])
    else:
        keep = "PASS" if len(wa) >= 2 and len(wa) - keep_hit <= 1 else "FAIL"
        drop = "PASS" if wr and drop_hit / len(wr) >= 0.80 else "FAIL"
        burden = "PASS" if words and flags / words <= 0.05 else "FAIL"
        if keep == drop == burden == "PASS":
            outcome = ("ALL PASS -> propose replacing the old reviewer's suggestions in review with these flags "
                       "(both readings shown, no suggested fix), plus a separate structure check for dividers")
        else:
            outcome = "NOT ALL PASS (failed: " + ", ".join(
                k for k, v in (("KEEP", keep), ("DROP", drop), ("BURDEN", burden)) if v != "PASS") + ")"
        print("\nCriteria (fixed 2026-11-17 before the first reading):")
        print(f"KEEP: {keep} ({keep_hit}/{len(wa)} approved word-level at a flag; at most one missed, at least 2 exist)")
        print(f"DROP: {drop} ({drop_hit}/{len(wr)} rejected word-level at no flag >= 80%?)")
        print(f"BURDEN: {burden} ({flags}/{words} flags per primary word <= 5%?)")
        print(f"OUTCOME: {outcome}")
        print("Limitation: one Opus band reading per page (one run).")
        verdict = {"keep": keep, "drop": drop, "burden": burden, "outcome": outcome}
    t, a = committed()
    RESULT.write_text(json.dumps({
        "pages": pages, "corrections": rows, "decisions_info": info, "verdict": verdict,
        "spend": {"counted": t, "settled": a}, "criteria": __doc__[__doc__.index("CRITERIA"):]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {RESULT}")
    return 0


def main() -> int:
    if CMD == "dry":
        return cmd_dry()
    if CMD == "read":
        return cmd_read("--sequential" in sys.argv)
    if CMD == "score":
        return cmd_score()
    print(__doc__.split("CRITERIA")[0])
    return 1


if __name__ == "__main__":
    sys.exit(main())
