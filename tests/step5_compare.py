#!/usr/bin/env python3
"""Plan Step 5: correction-level paired comparison, current engine vs new engine.

    ./venv/bin/python tests/step5_compare.py freeze            # population + current engine's boxes ($0)
    ./venv/bin/python tests/step5_compare.py read [--limit N]  # PAID: one Sonnet band reading per page
    ./venv/bin/python tests/step5_compare.py status
    ./venv/bin/python tests/step5_compare.py smoke             # $0: correction mapping checked on audited pages
    ./venv/bin/python tests/step5_compare.py build [--jobs 6]  # $0: new engine's box per correction; marking sample
    ./venv/bin/python tests/step5_compare.py serve [--port 8773] [--open]
    ./venv/bin/python tests/step5_compare.py report

Population (frozen before any new-engine box exists): every correction the
review UI derives (``review._page_box_inputs`` + ``review._box_specs``) on the
pages of ``ghessehaye-majid`` and ``bachehaye-ghali-fresh30-20260906-123256``
where the current engine has at least one cached line reading, so the current
engine is compared with its evidence, not without it. hadpi-articles has no
pending correction and is judged by its audited truth and the held-out
spot-check instead.

``freeze`` replays the current engine offline (dead API address) and records
its result per correction in ``out/step5/population.json`` (private: holds page
text). It refuses to run twice. ``read`` asks for one line-by-line reading per
page with arm ``t1r`` of ``tests/transcription_experiments.py`` unchanged
(Sonnet 5 on native bands), cached in that harness's cache, with its own ledger
``out/transcription_experiments/ledger_step5.jsonl`` and a hard cap.

New engine per page (frozen configuration, nothing tuned here): the reading's
printed lines -> L1 lines -> line-text assignment ``--la T`` -> arms, word rule
S∧T (printed pages, D13). A correction's Markdown span is mapped to the
reading's words by a page-wide word alignment on folded text; its box is the
union of its words' boxes when every one is located, a line-only box when the
lines are known, else nothing (``word_placer_eval.span_placed``). A correction
with no span is searched for in the reading by its text; an insertion takes
the word on each side of the insertion point.

Judging: no audited boxes exist on these pages, so the owner marks every shown
box of both engines, shuffled and unlabelled: RIGHT (the whole target is inside
and at most one extra word on each side), WIDE (the whole target is inside,
with more excess than that), MISS (right line, but part or all of the target is
outside), WRONG LINE, or cannot tell. "Covered" = right + wide (the plan's covered-target rate);
"correct" = right (plan section 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "out" / "step5"
POPULATION = OUT / "population.json"
SLUGS = {"m": "ghessehaye-majid", "f": "bachehaye-ghali-fresh30-20260906-123256"}
ARM = "t1r"
CAP_USD = 3.60
LEDGER = "ledger_step5.jsonl"
DEAD = "http://127.0.0.1:9"


def pid_of(slug: str, page: int) -> str:
    return next(k for k, v in SLUGS.items() if v == slug) + f"{page:04d}"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cmd_freeze(_args) -> None:
    if POPULATION.exists():
        raise SystemExit(f"{POPULATION} exists: the population is frozen")
    if os.environ.get("ANTHROPIC_BASE_URL") != DEAD:
        raise SystemExit(f"set ANTHROPIC_BASE_URL={DEAD} (freeze must not reach the network)")
    from farsi2epub import review
    from farsi2epub.scan import PlacementService
    from farsi2epub.workspace import Workspace
    pages = []
    for slug in SLUGS.values():
        ws = Workspace.load(slug)
        svc = PlacementService(ws, mode="offline")
        for f in sorted(ws.text_dir.glob("[0-9][0-9][0-9][0-9].json")):
            n = int(f.stem)
            _, issues, hunks, md, _ = review._page_box_inputs(ws, n)
            specs = review._box_specs(issues, hunks, md)
            if not specs:
                continue
            svc.ensure_page_map(n, allow_api=False)
            sm = svc.page_map_summary(n) or {}
            if not (sm.get("read_lines") or 0):
                continue
            qs = [s[1] for s in specs]
            svc.replay(n, md, qs, [None] * len(qs))
            res = svc.results(n, md, qs, [None] * len(qs))
            pages.append({
                "slug": slug, "page": n, "pid": pid_of(slug, n),
                "md_sha256": _sha(md.encode("utf-8")), "sidecar_sha256": _sha(f.read_bytes()),
                "map": {k: sm.get(k) for k in ("status", "detected_lines", "read_lines", "verified_words")},
                "corrections": [{"key": s[0], "kind": s[1].kind, "span": list(s[1].span) if s[1].span else None,
                                 "text": s[1].text, "current": r.to_dict()} for s, r in zip(specs, res)]})
            print(f"  {slug[:18]:18s} p{n:<4d} corrections {len(specs)}", flush=True)
        if svc.budget.spent:
            raise SystemExit("freeze spent money; refusing to write")
        svc.close()
    OUT.mkdir(parents=True, exist_ok=True)
    POPULATION.write_text(json.dumps({"version": "step5/1", "arm": ARM, "pages": pages}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    n = sum(len(p["corrections"]) for p in pages)
    print(f"froze {n} corrections on {len(pages)} pages -> {POPULATION}")


def population() -> dict:
    if not POPULATION.exists():
        raise SystemExit("run `freeze` first")
    return json.loads(POPULATION.read_text(encoding="utf-8"))


def _harness():
    import transcription_experiments as TE
    for p in population()["pages"]:
        TE.PAGES[p["pid"]] = (p["slug"], p["page"])
    return TE


def reading(TE, runner, pid: str):
    """Stitched printed lines of the cached reading, or None."""
    rec = runner.cached(TE.build_request(ARM, pid, 0))
    out = TE.output_of(rec, ARM) if rec is not None else None
    if out is None or out.get("lines") is None:
        return None
    return [l for l in out["lines"] if l and l.strip()]


def cmd_read(args) -> None:
    TE = _harness()
    runner = TE.Runner(cap=CAP_USD, ledger_name=LEDGER)
    todo = [p["pid"] for p in population()["pages"]]
    done = 0
    for pid in todo:
        req = TE.build_request(ARM, pid, 0)
        if runner.cached(req) is None:
            if args.limit is not None and done >= args.limit:
                continue
            done += 1
        try:
            rec = runner.call(req)
        except (TE.BudgetExceeded, TE.AccessError) as exc:
            print(f"STOP: {type(exc).__name__}: {exc}")
            break
        out = TE.output_of(rec, ARM)
        print(f"{pid} hit={rec['cache_hit']} in={rec['usage']['input_tokens']} out={rec['usage']['output_tokens']} "
              f"${rec['cost']:.4f} stop={rec.get('stop_reason')} lines={len(out.get('lines') or [])} "
              f"ledger ${runner.ledger.total():.4f}", flush=True)
    cmd_status(args)


def cmd_status(_args) -> None:
    TE = _harness()
    runner = TE.Runner(cap=CAP_USD, ledger_name=LEDGER)
    pages = population()["pages"]
    have = sum(reading(TE, runner, p["pid"]) is not None for p in pages)
    print(f"pages {len(pages)}  corrections {sum(len(p['corrections']) for p in pages)}  "
          f"readings cached {have}  ledger ${runner.ledger.total():.4f} of cap ${CAP_USD:.2f}")


# ---------------------------------------------------------------------------
# correction -> words of the reading
# ---------------------------------------------------------------------------
import difflib
import re

MIN_TEXT_MATCH = 0.75      # span-less corrections: similarity of the best window of the reading
MIN_TEXT_MARGIN = 0.10     # ... and its lead over the best non-overlapping window


def _fold(tok: str) -> str:
    from farsi2epub import locate
    return locate._fold_word(tok)


def md_tokens(md: str) -> list[tuple[int, int, str]]:
    """(start, end, folded) for every whitespace-separated token that folds to something."""
    return [(m.start(), m.end(), f) for m in re.finditer(r"\S+", md) if (f := _fold(m.group()))]


def align_words(a: list[str], b: list[str]) -> dict[int, list[int]]:
    """a index -> b indices: equal runs and equal-length replacements position by
    position; an unequal replacement maps each a word to the whole b block."""
    out: dict[int, list[int]] = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                out[i1 + k] = [j1 + k]
        elif tag == "replace":
            for i in range(i1, i2):
                out[i] = list(range(j1, j2))
    return out


def target_tokens(toks: list[tuple[int, int, str]], span, kind: str) -> list[int]:
    s, e = span
    hit = [i for i, (a, b, _) in enumerate(toks) if a < e and b > s] if e > s and kind != "insertion" else []
    if hit:
        return hit
    before = [i for i, (_, b, _) in enumerate(toks) if b <= s]
    after = [i for i, (a, _, _) in enumerate(toks) if a >= e]
    return ([before[-1]] if before else []) + ([after[0]] if after else [])


def find_text(text: str, rd: list[str]):
    """Reading indices of the best window for a span-less correction, or (None, reason)."""
    want = [f for w in text.split() if (f := _fold(w))]
    if not want:
        return None, "correction has no text and no span"
    joined, n = "".join(want), len(want)
    cands = []
    for size in {max(1, n - 1), n, n + 1}:
        for j in range(0, len(rd) - size + 1):
            r = difflib.SequenceMatcher(None, joined, "".join(rd[j:j + size]), autojunk=False).ratio()
            cands.append((r, -abs(size - n), j, size))
    if not cands:
        return None, "reading is empty"
    cands.sort(reverse=True)
    r, _, j, size = cands[0]
    rival = max((c[0] for c in cands if c[2] + c[3] <= j or c[2] >= j + size), default=0.0)
    if r < MIN_TEXT_MATCH:
        return None, f"text not found in the reading (best {r:.2f})"
    if r - rival < MIN_TEXT_MARGIN:
        return None, f"text occurs more than once in the reading ({r:.2f} vs {rival:.2f})"
    return list(range(j, j + size)), ""


def reading_keys(lines: list[list[str]]) -> tuple[list[tuple[int, int]], list[str]]:
    keys = [(li, ti) for li, l in enumerate(lines) for ti in range(len(l))]
    return keys, [_fold(lines[li][ti]) for li, ti in keys]


def place_correction(arm: dict, lines: list[list[str]], md: str, span, kind: str, text: str) -> dict:
    """The new engine's result for one correction: {status, segments, reason, tokens}."""
    import word_placer_eval as E
    keys, rd = reading_keys(lines)
    if span is None:
        idx, why = find_text(text, rd)
        if idx is None:
            return {"status": "unresolved", "segments": [], "reason": why, "tokens": []}
    else:
        toks = md_tokens(md)
        amap = align_words([t[2] for t in toks], rd)
        hit = sorted({j for i in target_tokens(toks, span, kind) for j in amap.get(i, [])})
        if not hit:
            return {"status": "unresolved", "segments": [], "reason": "target words have no counterpart in the reading",
                    "tokens": []}
        idx = list(range(hit[0], hit[-1] + 1))
    pl = E.span_placed([arm.get(keys[j]) for j in idx])
    if pl is None:
        return {"status": "unresolved", "segments": [], "reason": "a target word has no placement", "tokens": idx}
    return {"status": pl["status"], "segments": pl["segments"], "reason": pl.get("reason", ""), "tokens": idx}


RULE = "ST"


def run_engine(name: str, gray, lines: list[list[str]], truth_page=None) -> tuple[dict, dict]:
    import real_truth_eval as rt
    import word_placer as WP
    import word_placer_eval as E
    E.T_OPTS.update(vert="comp", la="T")
    H, W = gray.shape
    slug, page = name.rsplit("/", 1)
    tp = truth_page or rt.TruthPage(slug, int(page), "printed", "step5_no_boxes", W, H, Path(""), "", Path(""), "", [], [])
    ev = WP.page_evidence(gray)
    pg = dict(name=name, gray=gray, truth_page=tp, texts=lines, kind="step5")
    arms, info = E.run_T(pg, "C", ev=ev, texts=lines, src=None)
    return arms[RULE], info


# ---------------------------------------------------------------------------
# smoke: the mapping checked on the audited pages (cached readings, $0)
# ---------------------------------------------------------------------------
SMOKE = {"c23": ("ghessehaye-majid", 23), "c109": ("ghessehaye-majid", 109)}
SMOKE_PASS = {"shown_min": 0.85, "precision_min": 0.98, "skipped_max": 0.15}


def cmd_smoke(_args) -> None:
    if os.environ.get("ANTHROPIC_BASE_URL") != DEAD:
        raise SystemExit(f"set ANTHROPIC_BASE_URL={DEAD}")
    import random
    import real_truth_eval as rt
    import word_placer_e2e as X
    import word_placer_eval as E
    import word_placer_passage as PS
    lengths = [k for k, w in PS.SPAN_WEIGHTS.items() for _ in range(w)]
    tot = {"spans": 0, "skipped": 0, "shown": 0, "correct": 0, "line": 0, "unresolved": 0}
    for pid, (slug, n) in SMOKE.items():
        page = E.real_page(slug, n)
        tp = page["truth_page"]
        lines = [l.split() for l in X.model_lines(ARM, pid)]
        arm, _ = run_engine(f"{slug}/{n}", page["gray"], lines, truth_page=tp)
        md = page["md"]
        toks = md_tokens(md)
        tkeys = [(L.index, w.wi) for L in tp.lines if not L.excluded for w in L.words]
        tmap = align_words([t[2] for t in toks], [_fold(tp.word(*k).text) for k in tkeys])
        rnd = random.Random(E.hash_seed(page["name"], "step5-smoke"))
        row = dict.fromkeys(tot, 0)
        for _ in range(100):
            k = rnd.choice(lengths)
            i0 = rnd.randrange(0, max(1, len(toks) - k))
            ids = list(range(i0, min(len(toks), i0 + k)))
            row["spans"] += 1
            target = [tmap.get(i) for i in ids]
            if any(t is None or len(t) != 1 or _fold(tp.word(*tkeys[t[0]]).text) != toks[i][2] for t, i in zip(target, ids)):
                row["skipped"] += 1          # the Markdown differs from the truth here: no clean truth target
                continue
            pl = place_correction(arm, lines, md, (toks[ids[0]][0], toks[ids[-1]][1]), "replacement", "")
            if pl["status"] == "located":
                row["shown"] += 1
                row["correct"] += bool(rt.grade_target_h1(tp, [tkeys[t[0]] for t in target], pl)["correct"])
            else:
                row[pl["status"]] += 1
        print(f"  {slug}/{n}: {row}", flush=True)
        for k in tot:
            tot[k] += row[k]
    graded = tot["spans"] - tot["skipped"]
    shown, prec, skip = tot["shown"] / graded, tot["correct"] / max(1, tot["shown"]), tot["skipped"] / tot["spans"]
    ok = shown >= SMOKE_PASS["shown_min"] and prec >= SMOKE_PASS["precision_min"] and skip <= SMOKE_PASS["skipped_max"]
    print(f"spans {tot['spans']}  skipped {tot['skipped']} ({skip:.1%})  shown {tot['shown']}/{graded} = {shown:.1%}  "
          f"correct {tot['correct']}/{tot['shown']} = {prec:.1%}  line-only {tot['line']}  unresolved {tot['unresolved']}")
    print(f"criteria {SMOKE_PASS}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# build: the new engine's box per correction, then the marking sample
# ---------------------------------------------------------------------------
PAGES_DIR = OUT / "pages"
SAMPLE = OUT / "sample.json"
MARKS = OUT / "marks.json"
RESULT = OUT / "result.json"
VERDICTS = ("right", "wide", "miss", "wrong_line", "cannot_tell")
CONTEXT_CHARS = 70


def _atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def hires(slug: str, page: int) -> Path:
    return ROOT / "books" / slug / "pages" / "hires" / f"{page:04d}.png"


def _build_page(pid: str) -> str:
    os.environ["ANTHROPIC_BASE_URL"] = DEAD
    out = PAGES_DIR / f"{pid}.json"
    if out.exists():
        return f"{pid} cached"
    import line_finder as LF
    TE = _harness()
    p = next(x for x in population()["pages"] if x["pid"] == pid)
    md = (ROOT / "books" / p["slug"] / "text" / f"{p['page']:04d}.md").read_text(encoding="utf-8")
    if _sha(md.encode("utf-8")) != p["md_sha256"]:
        raise SystemExit(f"{pid}: the page text changed since the freeze")
    lines = reading(TE, TE.Runner(cap=CAP_USD, ledger_name=LEDGER), pid)
    if lines is None:
        raise SystemExit(f"{pid}: no cached reading (run `read`)")
    lines = [l.split() for l in lines]
    gray = LF.load_gray(hires(p["slug"], p["page"]))
    H, W = gray.shape
    failed = None
    try:
        arm, info = run_engine(f"{p['slug']}/{p['page']}", gray, lines)
    except Exception as exc:  # noqa: BLE001  the research engine failed on this page: it shows no box there
        arm, info, failed = {}, {}, f"engine failed on this page: {type(exc).__name__}: {exc}"
    rows = {}
    for c in p["corrections"]:
        pl = place_correction(arm, lines, md, c["span"], c["kind"], c["text"])
        if failed:
            pl = {"status": "unresolved", "segments": [], "reason": failed, "tokens": []}
        rows[c["key"]] = {"status": pl["status"], "reason": pl["reason"],
                          "quads": [[[round(a * W, 1), round(b * H, 1)] for a, b in sg["quad"]] for sg in pl["segments"]],
                          "reading_words": [lines[li][ti] for li, ti in (reading_keys(lines)[0][j] for j in pl["tokens"])]}
    words = sum(len(l) for l in lines)
    _atomic(out, {"pid": pid, "W": W, "H": H, "engine_failed": failed, "text_lines": len(lines), "words": words,
                  "words_located": sum(1 for x in arm.values() if x["status"] == "located"),
                  "line_assignment": info.get("la_template"),
                  "g2_boxes_on_nontext": info.get("g2_boxes_on_nontext"), "corrections": rows})
    return f"{pid}{' ENGINE FAILED' if failed else ''} lines {len(lines)} located {sum(1 for x in arm.values() if x['status'] == 'located')}/{words}"


def _display(md: str, c: dict) -> dict:
    if c["span"] is None:
        return {"before": "", "target": c["text"] or "(no text)", "after": "", "note": "finding without a position"}
    s, e = c["span"]
    return {"before": md[max(0, s - CONTEXT_CHARS):s], "target": md[s:e] if e > s else "▲",
            "after": md[e:e + CONTEXT_CHARS], "note": "insertion point (▲)" if c["kind"] == "insertion" or e == s else ""}


def cmd_build(args) -> None:
    if SAMPLE.exists():
        raise SystemExit(f"{SAMPLE} exists: the sample is built once")
    import random
    from concurrent.futures import ProcessPoolExecutor
    pop = population()
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for msg in ex.map(_build_page, [p["pid"] for p in pop["pages"]]):
            print("  " + msg, flush=True)
    items = []
    for p in pop["pages"]:
        b = json.loads((PAGES_DIR / f"{p['pid']}.json").read_text(encoding="utf-8"))
        md = (ROOT / "books" / p["slug"] / "text" / f"{p['page']:04d}.md").read_text(encoding="utf-8")
        W, H = b["W"], b["H"]
        for c in p["corrections"]:
            base = {"pid": p["pid"], "key": c["key"], "W": W, "H": H, **_display(md, c)}
            box = c["current"].get("box")
            if box:
                segs = box.get("segments") or [box]
                items.append({**base, "engine": "current", "status": c["current"]["status"],
                              "quads": [[[sg["x0"] * W, sg["y0"] * H], [sg["x1"] * W, sg["y0"] * H],
                                         [sg["x1"] * W, sg["y1"] * H], [sg["x0"] * W, sg["y1"] * H]] for sg in segs]})
            nb = b["corrections"][c["key"]]
            if nb["quads"]:
                items.append({**base, "engine": "new", "status": nb["status"], "quads": nb["quads"]})
    random.Random(20261114).shuffle(items)
    for k, it in enumerate(items):
        it["id"] = k
    _atomic(SAMPLE, {"schema": "step5_sample/1", "rule": RULE, "items": items})
    n = {e: sum(it["engine"] == e for it in items) for e in ("current", "new")}
    print(f"sample: {len(items)} boxes to mark ({n}) -> {SAMPLE}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _wilson(k: int, n: int, z: float = 1.96):
    import math
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def _mcnemar(b: int, c: int) -> float:
    """Exact two-sided p for b pairs won by one side and c by the other."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


def cmd_report(_args) -> None:
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    marks = json.loads(MARKS.read_text(encoding="utf-8")) if MARKS.exists() else {}
    items = sample["items"]
    print(f"Step 5, {len(marks)} of {len(items)} boxes marked")
    verdict = {(it["pid"], it["key"], it["engine"]): marks.get(str(it["id"])) for it in items}
    status = {(it["pid"], it["key"], it["engine"]): it["status"] for it in items}
    res = {"marked": len(marks), "boxes": len(items), "complete": len(marks) == len(items), "by": {}}
    pop = population()
    for scope in ("all",) + tuple(SLUGS):
        pairs = [(p["pid"], c["key"]) for p in pop["pages"] for c in p["corrections"] if scope in ("all", p["pid"][0])]
        row = {"corrections": len(pairs), "pages": len({a for a, _ in pairs})}
        outcome = {}
        for eng in ("current", "new"):
            v = [verdict.get((a, k, eng), "no_box") for a, k in pairs]
            cnt = {x: v.count(x) for x in VERDICTS + ("no_box", None)}
            outcome[eng] = v
            n = len(pairs) - cnt["cannot_tell"] - cnt[None]
            shown = len(pairs) - cnt["no_box"]
            row[eng] = {"shown": shown, "right": cnt["right"], "wide": cnt["wide"], "miss": cnt["miss"],
                        "wrong_line": cnt["wrong_line"], "cannot_tell": cnt["cannot_tell"], "unmarked": cnt[None],
                        "line_only": sum(status.get((a, k, eng)) == "line" for a, k in pairs),
                        "covered": cnt["right"] + cnt["wide"], "covered_rate_of_all": round((cnt["right"] + cnt["wide"]) / n, 4) if n else None,
                        "covered_95": _wilson(cnt["right"] + cnt["wide"], n),
                        "correct_rate_of_all": round(cnt["right"] / n, 4) if n else None, "correct_95": _wilson(cnt["right"], n),
                        "wrong_line_rate_of_all": round(cnt["wrong_line"] / n, 4) if n else None}
        for name, good in (("covered", ("right", "wide")), ("correct", ("right",))):
            usable = [(a, b) for a, b in zip(outcome["current"], outcome["new"])
                      if a not in ("cannot_tell", None) and b not in ("cannot_tell", None)]
            new_only = sum(b in good and a not in good for a, b in usable)
            cur_only = sum(a in good and b not in good for a, b in usable)
            row[f"paired_{name}"] = {"pairs": len(usable), "new_only": new_only, "current_only": cur_only,
                                     "both": sum(a in good and b in good for a, b in usable),
                                     "p_exact": round(_mcnemar(new_only, cur_only), 5)}
        res["by"][scope] = row
        label = "all" if scope == "all" else SLUGS[scope]
        print(f"\n{label}: {row['corrections']} corrections on {row['pages']} pages")
        for eng in ("current", "new"):
            r = row[eng]
            print(f"  {eng:7s} shown {r['shown']:3d}  right {r['right']:3d}  wide {r['wide']:3d}  miss {r['miss']:3d}  "
                  f"wrong line {r['wrong_line']:3d}  cannot tell {r['cannot_tell']}  unmarked {r['unmarked']}  "
                  f"line-only {r['line_only']}  covered {r['covered_rate_of_all']}  correct {r['correct_rate_of_all']}")
        for name in ("covered", "correct"):
            r = row[f"paired_{name}"]
            print(f"  paired {name}: new only {r['new_only']}, current only {r['current_only']}, both {r['both']}, "
                  f"of {r['pairs']} pairs; exact p {r['p_exact']}")
    if res["complete"]:
        _atomic(RESULT, res)
        print(f"\nwrote {RESULT}")
    else:
        print("\n(not complete: provisional, nothing written)")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
PAGE_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Step 5 box marking</title>
<style>
body{font:15px/1.45 -apple-system,system-ui,sans-serif;margin:0;background:#f4f2ee;color:#1d1d1f}
header{padding:10px 18px;background:#fff;border-bottom:1px solid #d8d4cc;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
main{padding:14px 18px;max-width:1500px;margin:auto}
.ctx{font:26px/1.7 "Vazirmatn","Geeza Pro",Tahoma,sans-serif;direction:rtl;text-align:center;margin:6px 0 4px;color:#777}
.ctx b{color:#111;background:#ffe9a8;border-radius:4px;padding:0 6px;font-weight:600;white-space:pre-wrap}
.note{color:#8a4b00;text-align:center;min-height:1.3em}
svg{display:block;width:100%;background:#fff;border:1px solid #d8d4cc;border-radius:6px;margin-bottom:10px}
.btns{display:flex;gap:10px;justify-content:center;margin:12px 0;flex-wrap:wrap}
button{font:inherit;padding:10px 18px;border-radius:8px;border:1px solid #9a958a;background:#fff;cursor:pointer}
button.on{outline:3px solid #1d1d1f}
#right{background:#dff3e2}#wide{background:#fff3c9}#miss{background:#f8dcd9}#wrong_line{background:#f3c6c1}#cannot_tell{background:#eee}
.rule{color:#555;max-width:1000px;margin:0 auto 6px;text-align:center}
kbd{border:1px solid #aaa;border-radius:4px;padding:0 5px;background:#fff}
</style></head><body>
<header><b>Step 5: is the box on the highlighted text?</b><span id="pos"></span><span id="done"></span>
<span>Keys: <kbd>R</kbd> right · <kbd>W</kbd> wide · <kbd>M</kbd> miss · <kbd>L</kbd> wrong line · <kbd>C</kbd> cannot tell · <kbd>←</kbd> <kbd>→</kbd></span></header>
<main>
<p class="rule">Find the highlighted text on the page (the stored text may differ slightly from the print: judge the printed place).
<b>Right</b>: all of it is inside the box, with at most one extra word on each side. <b>Wide</b>: all of it is inside, but the box takes more than that.
<b>Miss</b>: right line, but part or all of it is outside the box. <b>Wrong line</b>: the box is on another line. Height does not matter.
For an insertion point (▲) the box should sit at that place between the two words.</p>
<div class="ctx" id="ctx"></div><div class="note" id="note"></div>
<svg id="zoom" preserveAspectRatio="xMidYMid meet"></svg>
<svg id="strip" preserveAspectRatio="xMidYMid meet"></svg>
<div class="btns"><button id="right">Right (R)</button><button id="wide">Wide (W)</button><button id="miss">Miss (M)</button>
<button id="wrong_line">Wrong line (L)</button><button id="cannot_tell">Cannot tell (C)</button>
<button id="back">← Back</button><button id="next">Next →</button></div>
<p class="rule" id="finished"></p>
</main><script>
let S=null,i=0;const V=['right','wide','miss','wrong_line','cannot_tell'];
const $=id=>document.getElementById(id);
const esc=t=>t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function view(svg,it,x0,y0,w,h){
  x0=Math.max(0,x0);y0=Math.max(0,y0);w=Math.min(w,it.W-x0);h=Math.min(h,it.H-y0);
  svg.setAttribute('viewBox',`${x0} ${y0} ${w} ${h}`);svg.style.aspectRatio=`${w} / ${h}`;
  svg.innerHTML=`<image href="/img/${it.pid}.png" x="0" y="0" width="${it.W}" height="${it.H}"/>`+
   it.quads.map(q=>`<polygon points="${q.map(p=>p.join(',')).join(' ')}" fill="rgba(255,0,170,.10)" stroke="#e6007e" stroke-width="${Math.max(1.5,h/160)}"/>`).join('');
}
function show(){
  const it=S.items[i],pts=it.quads.flat(),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),lh=Math.max(30,(y1-y0)/Math.max(1,it.quads.length)),cx=(x0+x1)/2,cy=(y0+y1)/2;
  const zh=(y1-y0)+3*lh,zw=Math.max((x1-x0)+8*lh,Math.min(it.W,zh*4));
  view($('zoom'),it,cx-zw/2,cy-zh/2,zw,zh);
  view($('strip'),it,0,y0-3.5*lh,it.W,(y1-y0)+7*lh);
  $('ctx').innerHTML=esc(it.before)+'<b>'+esc(it.target)+'</b>'+esc(it.after);$('note').textContent=it.note||'';
  $('pos').textContent=`Box ${i+1} of ${S.items.length}`;
  const n=Object.keys(S.marks).length;$('done').textContent=`${n} marked`;
  const m=S.marks[it.id];V.forEach(v=>$(v).className=m===v?'on':'');
  $('finished').textContent=n===S.items.length?'All boxes are marked. You can close this page; tell Claude you are done.':'';
}
async function mark(v){
  const it=S.items[i];
  const r=await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:it.id,verdict:v})});
  if(!r.ok){$('finished').textContent='Save failed; is the server still running?';return}
  S.marks[it.id]=v;if(i<S.items.length-1)i++;show();
}
function go(d){i=Math.min(S.items.length-1,Math.max(0,i+d));show()}
V.forEach(v=>$(v).onclick=()=>mark(v));$('back').onclick=()=>go(-1);$('next').onclick=()=>go(1);
document.addEventListener('keydown',e=>{const k={r:'right',w:'wide',m:'miss',l:'wrong_line',c:'cannot_tell'}[e.key.toLowerCase()];
  if(k)mark(k);else if(e.key==='ArrowLeft')go(-1);else if(e.key==='ArrowRight')go(1)});
fetch('/api/state').then(r=>r.json()).then(s=>{S=s;const f=S.items.findIndex(it=>!(it.id in S.marks));i=f<0?0:f;show()});
</script></body></html>"""


def cmd_serve(args) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    if not SAMPLE.exists():
        raise SystemExit("run `build` first")
    pids = {p["pid"]: (p["slug"], p["page"]) for p in population()["pages"]}
    shown = ("id", "pid", "W", "H", "quads", "before", "target", "after", "note")   # never the engine or status

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if ctype == "image/png" else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/state":
                items = [{k: it[k] for k in shown} for it in json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]]
                marks = json.loads(MARKS.read_text(encoding="utf-8")) if MARKS.exists() else {}
                return self._send(200, json.dumps({"items": items, "marks": marks}, ensure_ascii=False).encode("utf-8"),
                                  "application/json")
            if path.startswith("/img/") and path.endswith(".png") and path[5:-4] in pids:
                return self._send(200, hires(*pids[path[5:-4]]).read_bytes(), "image/png")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/mark":
                return self._send(404, b"not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                ids = {it["id"] for it in json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]}
                if req["id"] not in ids or req["verdict"] not in VERDICTS:
                    raise ValueError("bad mark")
            except Exception as exc:  # noqa: BLE001
                return self._send(400, str(exc).encode(), "text/plain")
            marks = json.loads(MARKS.read_text(encoding="utf-8")) if MARKS.exists() else {}
            marks[str(req["id"])] = req["verdict"]
            _atomic(MARKS, marks)
            self._send(200, b"{}", "application/json")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Step 5 marking at {url} (Ctrl-C to stop)", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    srv.serve_forever()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    r = sub.add_parser("read")
    r.add_argument("--limit", type=int, default=None, help="at most N new paid calls")
    sub.add_parser("status")
    sub.add_parser("smoke")
    b = sub.add_parser("build")
    b.add_argument("--jobs", type=int, default=6)
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=8773)
    sv.add_argument("--open", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"freeze": cmd_freeze, "read": cmd_read, "status": cmd_status, "smoke": cmd_smoke, "build": cmd_build,
     "serve": cmd_serve, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
