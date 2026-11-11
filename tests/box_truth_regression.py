"""Review-time box feedback capture (feedback.py, docs/bbox-old-fonts-plan.md 6.3).

Synthetic and offline: builds its own two-page Persian workspace in a temp
dir, never reads books/, never calls the API, and writes only to a scratch
box_truth.jsonl. Exercises the REAL review HTTP handlers on 127.0.0.1.

    ./venv/bin/python tests/box_truth_regression.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fitz  # noqa: E402
import yaml  # noqa: E402

from farsi2epub import feedback, qc, review  # noqa: E402
from farsi2epub.scan import PlacementService  # noqa: E402
from farsi2epub.workspace import Workspace  # noqa: E402

FONT = ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf"
PAGES = {
    1: ["این کتاب درباره تاریخ ایران است", "و از روزگاران کهن سخن می گوید", "نویسنده آن مرد دانشمندی بود",
        "و همه عمر خود را صرف نوشتن کرد", "که سالها در شهر تهران زندگی کرد"],
    2: ["مردم شهر همیشه شاد بودند", "و هر روز به بازار می رفتند", "باغهای بزرگ در کنار رود",
        "میوه های فراوان به بار می آورد", "پرندگان روی درختان آواز خواندند"],
}
# (old -> new) word replacements per page, far enough apart to stay separate
# hunks; the empty-snippet issue on page 1 yields an i-key spec with no box.
EDITS = {1: [("تاریخ", "تواریخ"), ("دانشمندی", "دانشمند"), ("تهران", "طهران")],
         2: [("همیشه", "هميشه"), ("بزرگ", "بزرگی"), ("درختان", "درخت‌ها")]}


def make_workspace(books_root: Path, slug: str = "synthetic") -> Workspace:
    root = books_root / slug
    (root / "text").mkdir(parents=True)
    (root / "pages" / "hires").mkdir(parents=True)
    doc = fitz.open()
    for n in sorted(PAGES):
        page = doc.new_page(width=400, height=500)
        for i, line in enumerate(PAGES[n]):
            page.insert_text((40, 60 + i * 34), line, fontfile=str(FONT), fontname="vz", fontsize=16)
    doc.save(root / "source.pdf")
    for n in sorted(PAGES):
        doc[n - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(root / "pages" / "hires" / f"{n:04d}.png")
    doc.close()
    (root / "book.yaml").write_text(yaml.safe_dump({"title": "t", "page_count": 2, "source_type": "digital"}))
    for n, lines in PAGES.items():
        md = "\n".join(lines) + "\n"
        suggested = md
        issues = []
        for old, new in EDITS[n]:
            suggested = suggested.replace(old, new, 1)
            issues.append({"type": "wrong_word", "description": f"{old} misread", "snippet": old})
        if n == 1:
            issues.append({"type": "missing_text", "description": "line dropped", "snippet": ""})
        (root / "text" / f"{n:04d}.md").write_text(md, encoding="utf-8")
        (root / "text" / f"{n:04d}.json").write_text(json.dumps({
            "model_used": "synthetic", "confidence": 0.8, "quality_score": 0.9, "flags": [],
            "needs_review": True, "resolution": "hires",
            "qc": {"verdict": "fail", "suggestion_status": "pending", "suggested_text_md": suggested,
                   "issues": issues}}, ensure_ascii=False), encoding="utf-8")
    return Workspace(slug, books_root)


# -- HTTP helpers --------------------------------------------------------------

def call(base, path, body=None, raw=None, headers=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        text = e.read()
        try:
            return e.code, json.loads(text)
        except ValueError:
            return e.code, {"raw": text.decode("latin-1")}


def lines(path):
    return path.read_bytes().count(b"\n") if path.exists() else 0


class Server:
    def __init__(self, ws, pages, store, seed=12345, feedback_on=True):
        self.ws = ws
        review._REFINER = PlacementService(ws, mode="offline")
        self.state = review._ReviewState(ws, pages, [])
        if feedback_on:
            self.state.feedback = feedback.FeedbackSession(ws, pages, path=store, seed=seed)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), review._make_handler(self.state))
        self.state.httpd = self.httpd
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        review._start_refine_pipeline(self.state)
        for n in pages:  # wait until offline placement is terminal for every page
            deadline = time.time() + 30
            while review._boxes_payload(ws, n).get("pending"):
                assert time.time() < deadline, "placement stayed pending"
                time.sleep(0.05)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        review._REFINER.close()
        review._REFINER = None


# -- tests ---------------------------------------------------------------------

def test_split_rule():
    assert feedback.split_for("0" * 64) == feedback.split_for("0" * 64)
    # Pinned vectors: the rule must never silently change.
    v = {s: feedback.split_for(hashlib.sha256(s.encode()).hexdigest()) for s in ("a", "b", "c", "d")}
    expected = {s: ("heldout" if int(hashlib.sha256(("box-truth-split-v1:" + hashlib.sha256(s.encode()).hexdigest())
                                                      .encode()).hexdigest(), 16) % 2 else "dev") for s in v}
    assert v == expected, v
    rng = random.Random(1)
    shas = ["%064x" % rng.getrandbits(256) for _ in range(4000)]
    share = sum(feedback.split_for(s) == "heldout" for s in shas) / len(shas)
    assert 0.46 < share < 0.54, share
    assert all(feedback.split_for(s) == feedback.split_for(s) for s in shas[:100])
    assert feedback.SPLIT_RULE == "box-truth-image-sha256-parity-v1"


def test_sampling():
    entries = [("%064x" % i, f"c_{i:020d}", {"status": st}) for i, st in
               zip(range(40), ["located", "buffered", "estimated", "unresolved"] * 10)]
    pop, a = feedback.draw_sample(entries, 7, 5)
    shuffled = entries[:]
    random.Random(3).shuffle(shuffled)
    restatused = [(s, c, {"status": "unresolved"}) for s, c, _ in shuffled]
    _, b = feedback.draw_sample(restatused, 7, 5)
    assert [x[:2] for x in a] == [x[:2] for x in b], "sample depends on input order or status"
    assert len(a) == 5 and len({x[1] for x in a}) == 5
    assert [x[:2] for x in a] == [x[:2] for x in random.Random(7).sample(sorted(entries, key=lambda t: t[:2]), 5)]
    seeds = {tuple(x[1] for x in feedback.draw_sample(entries, s, 5)[1]) for s in range(20)}
    assert len(seeds) > 15, "different seeds should give different samples"
    counts = {e[1]: 0 for e in entries}
    for s in range(4000):
        for x in feedback.draw_sample(entries, s, 5)[1]:
            counts[x[1]] += 1
    expected = 4000 * 5 / 40
    assert all(abs(c - expected) < 0.25 * expected for c in counts.values()), counts
    assert feedback.draw_sample(entries[:3], 1, 5)[1] and len(feedback.draw_sample(entries[:3], 1, 5)[1]) == 3
    assert feedback.draw_sample([], 1, 5) == ([], [])


def test_store_concurrency_and_tolerance(tmp: Path):
    path = tmp / "conc.jsonl"

    def rec(tag, i):
        return {"schema": feedback.SCHEMA, "record_id": hashlib.md5(f"{tag}-{i}".encode()).hexdigest(),
                "action": "confirm", "family": "box_verdict", "pad": "ی" * 3000, "tag": tag, "i": i}

    def worker(tag):
        store = feedback.BoxTruthStore(path)
        for i in range(40):
            store.append(rec(tag, i))

    threads = [threading.Thread(target=worker, args=(f"t{k}",)) for k in range(8)]
    code = ("import sys; sys.path.insert(0, %r); from farsi2epub import feedback; import hashlib\n"
            "s = feedback.BoxTruthStore(%r)\n"
            "for i in range(40): s.append({'schema': feedback.SCHEMA, 'record_id': hashlib.md5(f'{sys.argv[1]}-{i}'.encode()).hexdigest(),"
            " 'action': 'confirm', 'family': 'box_verdict', 'pad': 'x' * 9000, 'tag': sys.argv[1], 'i': i})\n") % (str(ROOT), str(path))
    procs = [subprocess.Popen([sys.executable, "-c", code, f"p{k}"]) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(p.wait(60) == 0 for p in procs)
    records, bad = feedback.read_records(path)
    assert bad == 0 and len(records) == 400, (bad, len(records))
    assert len({r["record_id"] for r in records}) == 400
    for line in path.read_bytes().split(b"\n")[:-1]:
        json.loads(line)  # every physical line is one whole record
    # Malformed lines and a torn tail (simulated crash) never break readers,
    # and the next append terminates the torn line instead of gluing onto it.
    with path.open("ab") as f:
        f.write(b"not json\n[1,2]\n{\"schema\": \"other\"}\n{\"x\": NaN}\n{\"schema\": \"farsi2epub.box-truth/v1\", \"record_id\": \"torn")
    feedback.BoxTruthStore(path).append(rec("after", 0))
    records, bad = feedback.read_records(path)
    assert len(records) == 401 and bad == 5, (len(records), bad)
    assert records[-1]["tag"] == "after"
    assert feedback.read_records(tmp / "missing.jsonl") == ([], 0)


def test_effective_and_retraction():
    base = {"schema": feedback.SCHEMA, "image": {"sha256": "a" * 64}, "correction_key": "c_1"}
    r1 = {**base, "record_id": "1" * 32, "action": "confirm", "family": "box_verdict"}
    r2 = {**base, "record_id": "2" * 32, "action": "wrong_line", "family": "box_verdict"}
    r3 = {**base, "record_id": "3" * 32, "action": "spot_check", "family": "spot_check"}
    undo2 = {**base, "record_id": "4" * 32, "action": "retract", "family": "box_verdict", "retracts": "2" * 32}
    eff = feedback.effective([r1, r2, r3])
    assert eff[("a" * 64, "c_1", "box_verdict")] is r2 and eff[("a" * 64, "c_1", "spot_check")] is r3
    eff = feedback.effective([r1, r2, r3, undo2])
    assert eff[("a" * 64, "c_1", "box_verdict")] is r1, "undo restores the previous mark"
    undo1 = {**undo2, "record_id": "5" * 32, "retracts": "1" * 32}
    assert ("a" * 64, "c_1", "box_verdict") not in feedback.effective([r1, r2, undo2, undo1])


def test_http_round_trip(tmp: Path):
    books = tmp / "books"
    ws = make_workspace(books)
    store = tmp / "box_truth.jsonl"
    hist = tmp / "qc_history.json"
    srv = Server(ws, [1, 2], store)
    try:
        with patch.object(qc, "QC_HISTORY_PATH", hist):
            _http_cases(srv, ws, store)
    finally:
        srv.close()
    # Step 1 PASS criterion: replaying the store against the current engine,
    # offline, reproduces every recorded box (pre-edit Markdown via .orig.md).
    import box_truth_replay
    rows, meta = box_truth_replay.replay(store, books)
    assert meta["malformed_lines"] == 0 and rows
    box_rows = [r for r in rows if not r.get("no_query")]
    assert box_rows and not any(r["stale"] for r in rows), [r for r in rows if r["stale"]]
    assert all(r["reproduced"] for r in box_rows), [r for r in box_rows if not r["reproduced"]]
    human = [r for r in box_rows if "human_iou" in r]
    assert human and all(0 <= r["human_iou"] <= 1 and 0 <= r["human_coverage"] <= 1 for r in human)
    table, spot = box_truth_replay.summarize(rows)
    assert sum(v.get("right", 0) + v.get("wrong", 0) + v.get("skipped", 0) for v in spot.values()) == 3
    iou, cov = box_truth_replay.compare_to_human({"x0": .1, "y0": .1, "x1": .3, "y1": .2},
                                                 [{"x0": .2, "y0": .1, "x1": .4, "y1": .2}])
    assert abs(iou - 1 / 3) < 1e-9 and abs(cov - 0.5) < 1e-9, (iou, cov)


def _http_cases(srv, ws, store):
    base = srv.base
    st, fb1 = call(base, "/feedback/1")
    assert st == 200 and fb1["ok"], fb1
    png = (ws.root / "pages" / "hires" / "0001.png").read_bytes()
    assert fb1["image_sha256"] == hashlib.sha256(png).hexdigest()
    assert fb1["split"] == feedback.split_for(fb1["image_sha256"])
    corr = fb1["corrections"]
    boxed = [k for k, v in corr.items() if v["has_box"]]
    unboxed = [k for k, v in corr.items() if not v["has_box"]]
    assert len(boxed) >= 2 and unboxed, corr
    assert any(k.startswith("i") for k in unboxed), "expected an issue-only spec without a box"
    # Correction keys are the benchmark's case ids.
    import bbox_benchmark as bench
    _pages, cases = bench._collect(ws.books_root, [ws.slug])
    bench_ids = {c["case_id"] for c in cases if c["page_id"] == feedback.page_id(ws.slug, 1, hashlib.sha256(ws.pdf_path.read_bytes()).hexdigest())}
    assert bench_ids == {v["correction_key"] for k, v in corr.items() if k.startswith("h")}, (bench_ids, corr)
    payload = review._boxes_payload(ws, 1)

    def post(path, body):
        before = lines(store)
        st, out = call(base, path, body)
        return st, out, lines(store) - before

    k0, k1, ku = boxed[0], boxed[1], unboxed[0]
    shown0 = payload["hunk_boxes"][k0[1:]] if k0.startswith("h") else payload["issue_boxes"][k0[1:]]
    st, out, added = post("/feedback", {"page": 1, "key": k0, "correction_key": corr[k0]["correction_key"],
                                        "action": "confirm", "shown_box": shown0})
    assert st == 200 and added == 1 and out["effective"]["box_verdict"]["action"] == "confirm", out
    st, out, added = post("/feedback", {"page": 1, "key": k1, "action": "wrong_line"})
    assert st == 200 and added == 1, out
    rects = [{"x0": 0.1, "y0": 0.2, "x1": 0.4, "y1": 0.25}, {"x0": 0.6, "y0": 0.3, "x1": 0.9, "y1": 0.35}]
    st, out, added = post("/feedback", {"page": 1, "key": ku, "action": "drag_fix", "rects": rects})
    assert st == 200 and added == 1 and out["human_rects"] == rects, out
    drag_id = out["record_id"]
    st, out, added = post("/feedback", {"page": 1, "key": k1, "action": "drag_fix", "rects": rects[:1]})
    assert st == 200 and out["effective"]["box_verdict"]["action"] == "drag_fix"
    redo_id = out["record_id"]

    # Prior human rectangles come back on the next load, marked by correction.
    st, again = call(base, "/feedback/1")
    assert again["corrections"][ku]["human_rects"] == rects
    assert again["corrections"][k1]["human_rects"] == rects[:1]
    assert again["corrections"][k1]["effective"]["box_verdict"]["record_id"] == redo_id
    # Undo appends a retraction; the earlier wrong_line becomes current again.
    st, out, added = post("/feedback/undo", {"record_id": redo_id})
    assert st == 200 and added == 1 and out["effective"]["box_verdict"]["action"] == "wrong_line", out
    assert out["human_rects"] == []
    st, out, added = post("/feedback/undo", {"record_id": redo_id})
    assert st == 409 and added == 0
    st, out, added = post("/feedback/undo", {"record_id": drag_id})
    assert st == 200 and call(base, "/feedback/1")[1]["corrections"][ku]["human_rects"] == []

    # Spot check: seeded, drawn once, stable within the session.
    st, spot = call(base, "/feedback/spot")
    assert st == 200 and not spot["pending"] and spot["seed"] == 12345, spot
    assert spot == call(base, "/feedback/spot")[1]
    assert spot["sample_size"] == min(5, spot["population_size"]) and spot["sample_size"] == 5
    boxed_all = []
    for n in (1, 2):
        f = call(base, f"/feedback/{n}")[1]
        boxed_all += [(f["image_sha256"], v["correction_key"]) for v in f["corrections"].values() if v["has_box"]]
    assert spot["population_size"] == len(boxed_all)
    _, expect = feedback.draw_sample([t + (None,) for t in boxed_all], 12345, 5)
    assert [i["correction_key"] for i in spot["items"]] == [e[1] for e in expect]
    st, out, added = post("/feedback/spot", {"sample_index": 0, "verdict": "right"})
    assert st == 200 and added == 1, out
    st, out, added = post("/feedback/spot", {"sample_index": 1, "verdict": "wrong", "rects": rects[:1]})
    assert st == 200 and added == 1 and out["human_rects"] == rects[:1]
    st, out, added = post("/feedback/spot", {"sample_index": 2, "verdict": "skipped"})
    assert st == 200 and added == 1

    # Invalid input: 4xx and nothing written.
    bad_cases = [
        ("/feedback", {"page": 99, "key": k0, "action": "confirm"}, 404),
        ("/feedback", {"page": 1, "key": "h4242", "action": "confirm"}, 404),
        ("/feedback", {"page": 1, "key": k0, "action": "explode"}, 400),
        ("/feedback", {"page": "1", "key": k0, "action": "confirm"}, 400),
        ("/feedback", {"page": True, "key": k0, "action": "confirm"}, 400),
        ("/feedback", {"page": 1, "key": k0, "correction_key": "c_stale", "action": "confirm"}, 409),
        ("/feedback", {"page": 1, "key": ku, "action": "confirm"}, 409),
        ("/feedback", {"page": 1, "key": k0, "action": "confirm", "rects": rects}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix"}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix", "rects": [{"x0": -0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2}]}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix", "rects": [{"x0": 0.1, "y0": 0.1, "x1": 1.2, "y1": 0.2}]}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix", "rects": [{"x0": 0.3, "y0": 0.1, "x1": 0.2, "y1": 0.2}]}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix", "rects": [{"x0": True, "y0": 0.1, "x1": 0.2, "y1": 0.2}]}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix", "rects": [{"x0": 0.1, "y0": 0.1, "x1": 0.2}]}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix", "rects": [{"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2, "z": 1}]}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix", "rects": [rects[0]] * 17}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "drag_fix", "rects": "nope"}, 400),
        ("/feedback", {"page": 1, "key": k0, "action": "confirm", "shown_box": 3}, 400),
        ("/feedback/spot", {"sample_index": 99, "verdict": "right"}, 404),
        ("/feedback/spot", {"sample_index": 0, "verdict": "maybe"}, 400),
        ("/feedback/spot", {"sample_index": 0, "verdict": "right", "rects": rects[:1]}, 400),
        ("/feedback/undo", {"record_id": "z" * 32}, 400),
        ("/feedback/undo", {"record_id": "0" * 32}, 404),
        ("/feedback/nowhere", {}, 404),
    ]
    for path, body, want in bad_cases:
        st, out, added = post(path, body)
        assert st == want and added == 0 and out.get("ok") is False, (path, body, st, out)
    for raw, want in ((b'{"page": 1, "key": "%s", "action": "drag_fix", "rects": [{"x0": NaN, "y0": 0.1, "x1": 0.2, "y1": 0.2}]}' % k0.encode(), 400),
                      (b'{"page": 1, "x": Infinity}', 400), (b"not json", 400), (b"[1, 2]", 400),
                      (b"\xff\xfe", 400), (b'{"pad": "' + b"x" * (70 * 1024) + b'"}', 413)):
        before = lines(store)
        st, out = call(base, "/feedback", raw=raw)
        assert st == want and lines(store) == before, (raw[:40], st, out)
    for path in ("/feedback/99", "/feedback/abc", "/feedback/1/2"):
        assert call(base, path)[0] == 404

    # Record contract.
    records, bad = feedback.read_records(store)
    assert bad == 0
    for r in records:
        errs = feedback.validate_record(r)
        assert not errs, (r["action"], errs)
        assert r["book"] == "synthetic" and r["split_rule"] == feedback.SPLIT_RULE
        assert r["image"]["sha256"] in {hashlib.sha256((ws.root / "pages" / "hires" / f"{n:04d}.png").read_bytes()).hexdigest() for n in (1, 2)}
        for rect in r["human"]["rects"]:
            assert all(0 <= rect[k] <= 1 for k in ("x0", "y0", "x1", "y1"))
        if r["action"] != "retract":
            assert r["engine"]["derivation_version"] is not None and r["engine"]["bbox_mode"] == "offline"
            assert r["shown"] is not None and "status" in r["shown"] and "reason" in r["shown"]
            assert r["target"]["query"]["text"] == r["target"]["old"] or r["view_key"].startswith("i")
            assert isinstance(r["page_conditions"]["page_map"], dict)
    by_action = {}
    for r in records:
        by_action.setdefault(r["action"], []).append(r)
    assert set(by_action) == {"confirm", "wrong_line", "drag_fix", "retract", "spot_check"}, set(by_action)
    confirm = by_action["confirm"][0]
    assert confirm["shown"]["client_agrees"] is True and confirm["shown"]["box"] == shown0
    chip = payload["hunk_scores"][k0[1:]] if k0.startswith("h") else payload["issue_scores"][k0[1:]]
    assert confirm["shown"]["status"] == chip["status"] and confirm["shown"]["source"] == chip["source"]
    assert confirm["shown"]["reason"] == chip["reason"] and confirm["shown"]["label"] == chip["label"]
    assert confirm["engine"]["page_key"] == review._REFINER._page_key(1)
    assert {r["spot_check"]["verdict"] for r in by_action["spot_check"]} == {"right", "wrong", "skipped"}
    assert all(r["spot_check"]["seed"] == 12345 for r in by_action["spot_check"])
    no_box = [r for r in by_action["drag_fix"] if r["view_key"] == ku][0]
    assert no_box["shown"]["box"] is None and no_box["shown"]["status"] == "unresolved"
    # Feedback observed and did not change any box.
    assert review._boxes_payload(ws, 1) == payload

    # Accept links text edits; Accept itself and qc_history stay unchanged.
    accept_text_edit(srv, ws, store, base)


def accept_text_edit(srv, ws, store, base):
    md = ws.page_md_path(1).read_text(encoding="utf-8")
    _sc, issues, hunks, _t, _k = review._page_box_inputs(ws, 1)
    real = [h for h in hunks if not h.get("edit_only")]
    approved, edited = real[0], real[1]
    new_text = md.replace(approved["old"], approved["new"], 1).replace(edited["old"], "ویرایش‌شده", 1)
    new_text = new_text.replace("کتاب", "کتابی", 1)  # free-form edit outside every hunk
    decisions = [{"id": h["id"], "decision": "pending"} for h in hunks]
    decisions[hunks.index(approved)]["decision"] = "approved"
    decisions[hunks.index(edited)] = {"id": edited["id"], "decision": "edited", "text": "ویرایش‌شده"}
    body = {"page": 1, "text": new_text, "hunks": decisions}

    # Control: an identical workspace with feedback disabled.
    control_root = Path(tempfile.mkdtemp(prefix="box-truth-control-"))
    try:
        cws = make_workspace(control_root / "books")
        chist = control_root / "qc_history.json"
        before = lines(store)
        st, out = call(base, "/accept", body)
        assert st == 200 and out["ok"], out
        srv_records = feedback.read_records(store)[0][-(lines(store) - before):]
        hist_with = json.loads(qc.QC_HISTORY_PATH.read_text())["events"]
        side_with = json.loads(ws.page_meta_path(1).read_text())
        md_with = ws.page_md_path(1).read_text(encoding="utf-8")
        old_refiner = review._REFINER
        try:
            with patch.object(qc, "QC_HISTORY_PATH", chist):
                control = Server(cws, [1, 2], control_root / "unused.jsonl", feedback_on=False)
                try:
                    cst, cout = call(control.base, "/accept", body)
                    assert call(control.base, "/feedback/1")[0] == 404
                finally:
                    control.close()
                hist_without = json.loads(chist.read_text())["events"]
        finally:
            review._REFINER = old_refiner
        assert not (control_root / "unused.jsonl").exists()
        assert (cst, cout) == (st, out), (cout, out)
        strip = lambda evs: [{k: v for k, v in e.items() if k != "date"} for e in evs]
        assert strip(hist_with) == strip(hist_without) and hist_with, "qc_history output changed"
        assert json.loads(cws.page_meta_path(1).read_text()) == side_with
        assert cws.page_md_path(1).read_text(encoding="utf-8") == md_with == new_text
    finally:
        shutil.rmtree(control_root, ignore_errors=True)

    edits = [r for r in srv_records if r["action"] == "text_edit"]
    assert len(edits) == 3 == len(srv_records), [r["text_edit"] for r in srv_records]
    by = {r["text_edit"]["decision"]: r for r in edits}
    assert by["approved"]["text_edit"]["old"] == approved["old"] and by["approved"]["text_edit"]["new"] == approved["new"]
    assert by["edited"]["text_edit"]["new"] == "ویرایش‌شده"
    assert by["free_edit"]["text_edit"]["old"].startswith("کتاب") and by["free_edit"]["correction_key"].startswith("t_")
    for r in edits:
        assert not feedback.validate_record(r), feedback.validate_record(r)
        assert r["markdown_sha256"] == hashlib.sha256(md.encode()).hexdigest()  # pre-edit identity
        s, e = r["text_edit"]["span"]
        assert md[s:e] == r["text_edit"]["old"]
    assert by["approved"]["shown"]["box"] is not None or by["approved"]["shown"]["status"] == "unresolved"
    # A no-change accept writes no feedback.
    before = lines(store)
    assert call(base, "/accept", {"page": 2, "text": ws.page_md_path(2).read_text(encoding="utf-8")})[0] == 200
    assert lines(store) == before


# -- coordinate maths: one pure JS function, checked in node and mirrored here --

def py_screen_to_page(cx, cy, r):
    if not r or not r["width"] > 0 or not r["height"] > 0:
        return None
    c = lambda v: 0 if v < 0 else 1 if v > 1 else v
    return {"x": c((cx - r["left"]) / r["width"]), "y": c((cy - r["top"]) / r["height"])}


def client_rect(wrap_left, wrap_top, iw, ih, s, tx, ty):
    """getBoundingClientRect() of an element with transform translate(tx,ty) scale(s), origin 0 0."""
    return {"left": wrap_left + tx, "top": wrap_top + ty, "width": iw * s, "height": ih * s}


COORD_CASES = []
for (s, tx, ty) in ((1, 0, 0), (2.5, -300, -120), (4, -1800, -2400), (1.37, -12.5, -3.25)):
    for wl, wt in ((0, 0), (37, 412)):  # RTL layout can offset the wrap inside its viewport
        r = client_rect(wl, wt, 640, 905, s, tx, ty)
        for px, py in ((0, 0), (1, 1), (0.25, 0.75), (0.5, 0.123)):
            COORD_CASES.append((r, r["left"] + px * r["width"], r["top"] + py * r["height"], px, py))
        COORD_CASES.append((r, r["left"] - 50, r["top"] + r["height"] + 9, 0, 1))  # clamped outside


def test_coordinates_python():
    for r, cx, cy, px, py in COORD_CASES:
        p = py_screen_to_page(cx, cy, r)
        assert abs(p["x"] - px) < 1e-9 and abs(p["y"] - py) < 1e-9, (r, cx, cy, p)
    assert py_screen_to_page(1, 1, {"left": 0, "top": 0, "width": 0, "height": 5}) is None


def test_template_and_js(tmp: Path):
    ws = make_workspace(tmp / "jsbooks")
    review._REFINER = None
    try:
        html = review._render_index(ws, [1, 2], [], bbox_refine=False)
    finally:
        review._REFINER = None
    assert 'id="fb-toolbar-1"' in html and 'id="fb-spot-btn"' in html
    assert html.count('class="fb-row"') == sum(1 for _ in re.finditer(r'class="hunk-item', html)) + 1
    body = html.split("</style>")[1].split("<script")[0]
    assert "qc-box-estimated" not in body
    node = shutil.which("node")
    if not node:
        print("  (node not found: JS syntax and pure-function checks SKIPPED)")
        return "skipped"
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    js = tmp / "review.js"
    js.write_text(scripts[0], encoding="utf-8")
    res = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    pure = scripts[0].split("// FB-PURE-BEGIN")[1].split("// FB-PURE-END")[0]
    cases = [{"r": r, "cx": cx, "cy": cy, "px": px, "py": py} for r, cx, cy, px, py in COORD_CASES]
    harness = pure + "\nvar cases = " + json.dumps(cases) + ";\n" + r"""
var bad = [];
cases.forEach(function (c) {
  var p = fbScreenToPage(c.cx, c.cy, c.r);
  if (Math.abs(p.x - c.px) > 1e-9 || Math.abs(p.y - c.py) > 1e-9) bad.push([c, p]);
});
if (fbScreenToPage(1, 1, {left: 0, top: 0, width: 0, height: 4}) !== null) bad.push('zero width');
var r = fbRectFromPoints({x: .6, y: .4}, {x: .2, y: .5}, FB_MIN_SIDE);
if (!(r.x0 === .2 && r.x1 === .6 && r.y0 === .4 && r.y1 === .5)) bad.push(['normalize', r]);
if (fbRectFromPoints({x: .5, y: .5}, {x: .501, y: .9}, FB_MIN_SIDE) !== null) bad.push('tiny width kept');
if (fbRectFromPoints({x: .5, y: .5}, {x: .5, y: .5}, 0) === null) bad.push('live zero rect dropped');
console.log(JSON.stringify(bad));
"""
    hp = tmp / "harness.js"
    hp.write_text(harness, encoding="utf-8")
    res = subprocess.run([node, str(hp)], capture_output=True, text=True)
    assert res.returncode == 0 and res.stdout.strip() == "[]", (res.stdout, res.stderr)
    return "ok"


def main():
    with tempfile.TemporaryDirectory(prefix="box-truth-test-") as d:
        tmp = Path(d)
        real_store = ROOT / "box_truth.jsonl"
        real_before = real_store.read_bytes() if real_store.exists() else None
        test_split_rule()
        test_sampling()
        test_effective_and_retraction()
        test_store_concurrency_and_tolerance(tmp)
        test_coordinates_python()
        js = test_template_and_js(tmp)
        test_http_round_trip(tmp)
        real_after = real_store.read_bytes() if real_store.exists() else None
        assert real_before == real_after, "test wrote to the real box_truth.jsonl"
    print(f"BOX TRUTH REGRESSIONS PASSED (js checks: {js})")


if __name__ == "__main__":
    main()
