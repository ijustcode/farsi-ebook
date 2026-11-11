#!/usr/bin/env python3
"""Replay box_truth.jsonl against the CURRENT placement engine, offline.

For every effective record (latest per image sha, correction key and action
family; retracted records excluded) this re-runs the current engine for that
correction exactly as review would show it (locate tier, then the offline
PlacementService replay, keeping only match/scan geometry), and reports per
split and per action:

  * how many shown boxes the current engine reproduces identically;
  * for records with human rectangles, how the current box compares to them
    (IoU of the unions, and the share of the human area the box covers,
    computed with tests/bbox_metrics.coverage);
  * stale records (book, page image, PDF or Markdown no longer match) are
    counted as stale, not errors.

Headline precision uses SPOT-CHECK records only; drag/confirm/wrong-line
records are selected by the reviewer and are biased. No API calls: the
service runs in offline mode and is never given a reader. Like
tests/bbox_capture.py, an offline replay may write derived box caches into a
book's locate_evidence/ when a result was not cached yet.

    ./venv/bin/python tests/box_truth_replay.py [--store box_truth.jsonl] [--books-root books] [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import bbox_metrics  # noqa: E402
from farsi2epub import feedback, locate, review  # noqa: E402
from farsi2epub.config import MODEL_STRONG, PRICES  # noqa: E402
from farsi2epub.scan import PlacementService  # noqa: E402
from farsi2epub.workspace import DEFAULT_BOOKS_ROOT, Workspace  # noqa: E402


def _rects(box):
    if not box:
        return []
    return [[s[k] for k in ("x0", "y0", "x1", "y1")] for s in (box.get("segments") or [box])]


def _union_area(rects):
    if not rects:
        return 0.0
    env = [min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)]
    return bbox_metrics.coverage(env, rects) * bbox_metrics.area(env)


def compare_to_human(box, human):
    """(IoU of unions, share of human area covered by the box)."""
    hr = [[r[k] for k in ("x0", "y0", "x1", "y1")] for r in human]
    br = _rects(box)
    h_area = _union_area(hr)
    if not br or h_area <= 0:
        return 0.0, 0.0
    # Intersection of the two unions = area of human union covered by box union.
    inter = _union_area_clipped(hr, br)
    union = _union_area(br) + h_area - inter
    return (inter / union if union > 0 else 0.0), inter / h_area


def _union_area_clipped(a, b):
    clips = []
    for x in a:
        for y in b:
            r = [max(x[0], y[0]), max(x[1], y[1]), min(x[2], y[2]), min(x[3], y[3])]
            if r[0] < r[2] and r[1] < r[3]:
                clips.append(r)
    return _union_area(clips)


def _sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _page_markdown(ws, n, want_sha):
    for path in (ws.page_md_path(n), ws.page_orig_path(n)):
        if path.is_file():
            md = path.read_text(encoding="utf-8")
            if hashlib.sha256(md.encode("utf-8")).hexdigest() == want_sha:
                return md
    return None


def replay(store, books_root):
    records, malformed = feedback.read_records(store)
    eff = [r for r in feedback.effective(records).values()]
    retractions = sum(r["action"] == "retract" for r in records)
    groups = defaultdict(list)
    for r in eff:
        groups[(r["book"], r["page"], r["markdown_sha256"], r["image"]["sha256"], r["pdf_sha256"])].append(r)
    rows = []
    services = {}
    try:
        for (book, n, md_sha, img_sha, pdf_sha), recs in sorted(groups.items(), key=lambda kv: kv[0][:2]):
            ws = Workspace(book, books_root)
            stale = None
            if not ws.pdf_path.is_file():
                stale = "book_missing"
            elif _sha_file(ws.pdf_path) != pdf_sha:
                stale = "pdf_changed"
            else:
                img = review._image_path_for(ws, n)
                if img is None or feedback.sha256_file(img) != img_sha:
                    stale = "image_changed"
            md = None if stale else _page_markdown(ws, n, md_sha)
            if md is None and not stale:
                stale = "markdown_changed"
            replayable = [r for r in recs if (r.get("target") or {}).get("query")]
            current = {}
            if not stale and replayable:
                queries = [locate.Query(q["text"], tuple(q["span"]) if q["span"] else None,
                                        tuple(q["alternatives"]), q["kind"])
                           for q in (r["target"]["query"] for r in replayable)]
                model = (replayable[0].get("engine") or {}).get("reader_model") or MODEL_STRONG
                model = model if model in PRICES else MODEL_STRONG
                key = (book, model)
                if key not in services:
                    services[key] = PlacementService(ws, model, mode="offline")
                svc = services[key]
                initial = locate.locate_queries(ws.pdf_path, n, md, queries)
                boxes = svc.replay(n, md, queries, initial)
                for r, b in zip(replayable, boxes):
                    current[r["record_id"]] = b if b and b.get("source") in {"match", "scan"} else None
            for r in recs:
                row = {"record_id": r["record_id"], "action": r["action"], "split": r["split"],
                       "book": book, "page": n, "stale": stale}
                if not stale and r["record_id"] in current:
                    shown = (r.get("shown") or {})
                    cur = current[r["record_id"]]
                    row["shown_pending"] = shown.get("status") == "pending"
                    row["reproduced"] = feedback._boxes_equal(shown.get("box"), cur)
                    row["current_has_box"] = cur is not None
                    human = (r.get("human") or {}).get("rects") or []
                    if human:
                        iou, cov = compare_to_human(cur, human)
                        row["human_iou"], row["human_coverage"] = iou, cov
                elif not stale:
                    row["no_query"] = True
                if r["action"] == "spot_check":
                    row["verdict"] = r["spot_check"]["verdict"]
                rows.append(row)
    finally:
        for svc in services.values():
            svc.close()
    return rows, {"records": len(records), "malformed_lines": malformed, "retractions": retractions,
                  "effective": len(eff)}


def summarize(rows):
    table = defaultdict(lambda: defaultdict(float))
    for r in rows:
        for key in ((r["split"], r["action"]), (r["split"], "ALL")):
            t = table[key]
            t["n"] += 1
            if r["stale"]:
                t["stale"] += 1
                continue
            if r.get("no_query"):
                t["no_box_to_replay"] += 1
                continue
            t["replayed"] += 1
            t["reproduced"] += r["reproduced"]
            t["shown_pending"] += r["shown_pending"]
            if "human_iou" in r:
                t["with_human_rects"] += 1
                t["iou_sum"] += r["human_iou"]
                t["cov_sum"] += r["human_coverage"]
                t["iou_ge_0_5"] += r["human_iou"] >= 0.5
                t["current_no_box"] += not r["current_has_box"]
    spot = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["action"] == "spot_check":
            s = spot[r["split"]]
            s[r["verdict"]] += 1
            if r["verdict"] in ("right", "wrong") and not r["stale"] and r.get("reproduced"):
                s["judged_and_reproduced"] += 1
    return table, spot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=feedback.store_path())
    ap.add_argument("--books-root", type=Path, default=DEFAULT_BOOKS_ROOT)
    ap.add_argument("--json", type=Path, default=None, help="also write per-record rows + summary")
    args = ap.parse_args(argv)
    rows, meta = replay(args.store, args.books_root)
    table, spot = summarize(rows)
    print(f"store: {args.store}")
    print(f"records: {meta['records']} ({meta['retractions']} retractions, {meta['malformed_lines']} malformed lines skipped); "
          f"effective: {meta['effective']}")
    print("\nreproduction of the shown box by the CURRENT engine (offline)")
    print(f"{'split':8} {'action':11} {'n':>4} {'stale':>5} {'replay':>6} {'same':>5} {'pend':>4} {'human':>5} {'meanIoU':>7} {'meanCov':>7} {'IoU>=.5':>7} {'noBox':>5}")
    for (split, action) in sorted(table):
        t = table[(split, action)]
        h = t["with_human_rects"]
        mi = f"{t['iou_sum'] / h:.3f}" if h else "-"
        mc = f"{t['cov_sum'] / h:.3f}" if h else "-"
        print(f"{split:8} {action:11} {int(t['n']):4} {int(t['stale']):5} {int(t['replayed']):6} {int(t['reproduced']):5} "
              f"{int(t['shown_pending']):4} {int(h):5} {mi:>7} {mc:>7} {int(t['iou_ge_0_5']):7} {int(t['current_no_box']):5}")
    print("\nHEADLINE PRECISION: spot-check records only (the only unbiased sample).")
    print("Drag-to-fix, confirm and wrong-line records are reviewer-selected and are never used for rates.")
    for split in ("dev", "heldout"):
        s = spot.get(split, {})
        judged = s.get("right", 0) + s.get("wrong", 0)
        prec = f"{s.get('right', 0) / judged:.3f}" if judged else "n/a"
        print(f"  {split:8} right={s.get('right', 0)} wrong={s.get('wrong', 0)} skipped={s.get('skipped', 0)} "
              f"precision(shown engine)={prec}; judged boxes the current engine reproduces: {s.get('judged_and_reproduced', 0)}/{judged}")
    print("  Gates use heldout only; tuning uses dev only. Precision applies to the engine version that showed the box;"
          " it carries over to the current engine only for reproduced boxes.")
    if args.json:
        args.json.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
