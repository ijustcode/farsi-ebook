#!/usr/bin/env python3
"""Box-audit packet: data contract, drafting, save/resume, undo, the
"page audited" rule and the HTTP tool (tests/box_audit.py).

Synthetic and offline: builds a one-page workspace from tests/synth_pages.py
in a temp dir, never reads books/, never calls the API.

    ./venv/bin/python tests/box_audit_regression.py
"""
from __future__ import annotations

import copy
import json
import socket
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import fitz  # noqa: E402
import numpy as np  # noqa: E402

import bbox_annotation_workflow as wf  # noqa: E402
import box_audit as ba  # noqa: E402
import box_audit_draft as bd  # noqa: E402
import synth_pages as sp  # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def make_workspace(tmp: Path, slug="synthbook", page=1, seed=3, preset="photo"):
    spec = sp.make_spec("typewriter", preset=preset, seed=seed)
    g = sp.generate(spec)
    u8, truth = g["image"], g["truth"]
    H, W = u8.shape
    ws = tmp / "books" / slug
    (ws / "pages" / "hires").mkdir(parents=True)
    (ws / "truth").mkdir()
    png = ws / "pages" / "hires" / f"{page:04d}.png"
    fitz.Pixmap(fitz.csGRAY, W, H, np.ascontiguousarray(u8).tobytes(), False).save(str(png))
    doc = fitz.open()
    pg = doc.new_page(width=W / 3.06, height=H / 3.06)
    pg.insert_image(pg.rect, filename=str(png))
    doc.save(str(ws / "source.pdf"))
    lines = [l["text"] for l in truth["lines"]]
    (ws / "truth" / f"{page:04d}.lines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return u8, truth


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        u8, truth = make_workspace(root)

        # ---------------------------------------------------------- drafting
        out = ba.draft_page(root, "synthbook", 1)
        draft = json.loads(out.read_text(encoding="utf-8"))
        check(ba.validate_page(draft) == [], "draft passes the data contract")
        check(draft["schema_version"] == ba.SCHEMA_VERSION and draft["kind"] == "draft", "schema and kind recorded")
        check(len(draft["image"]["sha256"]) == 64 and len(draft["text_source"]["sha256"]) == 64,
              "image and text source SHA-256 recorded")
        check(draft["text_source"]["kind"] == "owner_text_truth_lines", "text source kind recorded")
        check(draft["tool_version"] == ba.TOOL_VERSION and draft["draft"]["draft_version"] == bd.DRAFT_VERSION,
              "tool and draft versions recorded")
        check(len(draft["lines"]) == len(truth["lines"]), "one drafted line per printed line")
        check(all(len(l["words"]) == len(l["tokens"]) for l in draft["lines"]), "one box per truth word")
        check(all(w["state"] == "draft" and "uncertain" in w for l in draft["lines"] for w in l["words"]),
              "every word starts as draft with an uncertain flag")
        w0 = draft["lines"][0]["words"][0]
        check(len(w0["quad"]) == 4 and len(w0["rect"]) == 4, "words carry a quad and its bounds")
        try:
            ba.draft_page(root, "synthbook", 1)
            check(False, "refuses to overwrite an untouched draft without --force")
        except ba.AuditError:
            check(True, "refuses to overwrite an untouched draft without --force")
        try:
            ba.draft_page(root, "hadpi-articles", 4)
            check(False, "held-out page refused")
        except ba.AuditError as e:
            check("held out" in str(e), "held-out page refused")
        wf_errs = wf.validate_lines(draft["page_id"], ba.to_workflow_lines(draft))
        check(wf_errs == [], f"draft exports to the annotation-workflow line shape ({wf_errs[:2]})")

        # draft accuracy on this synthetic page (synthetic pages are easier than real ones)
        r = ba.synth_eval_page(u8, truth)
        rate = r["ok"] / r["words"]
        print(f"     synthetic typewriter photo page: {r['ok']}/{r['words']} words = {rate:.3f}, "
              f"wrong and unflagged {r['wrong_unflagged']}, uncertain lines {r['uncertain_lines']}")
        check(rate >= 0.90, "draft places >= 90% of words on the synthetic photo page")
        check(r["wrong_unflagged"] <= max(3, 0.02 * r["words"]), "few wrong words are left unflagged")

        # ---------------------------------------------------------- markdown
        md = "# سرآغاز\n\nیک دو **سه** چهار[^1].\n\n```verse\nالف ب --- پ ت\n```\n\nپنج شش"
        paras = bd.markdown_paragraphs(md)
        check([p["tokens"] for p in paras] == [["سرآغاز"], ["یک", "دو", "سه", "چهار."], ["الف", "ب"], ["پ", "ت"], ["پنج", "شش"]],
              "markdown paragraphs: markup removed, hemistichs are separate")
        check(bd.token_parts(["می‌رود", "و"])[2] == [True, False], "ZWNJ parts give optional boundaries")

        # ---------------------------------------------------------- audit ops
        draft_sha = ba.file_sha(out)
        store = ba.AuditStore(root)
        doc = store.get("synthbook", 1)
        check(doc["kind"] == "audit" and doc["status"] == "draft" and doc["draft_file_sha256"] == draft_sha,
              "audit file created from the draft on first open")
        n = len(doc["lines"])
        li = next(i for i, l in enumerate(doc["lines"]) if len(l["words"]) >= 4)
        doc = store.act("synthbook", 1, li, "accept_line", {})
        L = doc["lines"][li]
        check(L["state"] == "accepted" and all(w["state"] == "accepted" for w in L["words"]),
              "accept line: line and words accepted")
        check(doc["status"] == "in_progress", "page in progress after the first accept")
        li2 = next(i for i, l in enumerate(doc["lines"]) if len(l["words"]) >= 4 and i != li)
        a, b = doc["lines"][li2]["words"][0]["box"], doc["lines"][li2]["words"][1]["box"]
        u = (a[0] + b[2]) / 2 + 2
        doc = store.act("synthbook", 1, li2, "set_boundary", {"word": 0, "u": u})
        L = doc["lines"][li2]
        check(abs(L["words"][0]["box"][0] - L["words"][1]["box"][2]) < 13 and L["words"][0]["state"] == "fixed"
              and L["words"][1]["state"] == "fixed" and L["state"] == "editing", "move a word boundary")
        wb = L["words"][2]["box"]
        doc = store.act("synthbook", 1, li2, "split", {"word": 2, "u": (wb[0] + wb[2]) / 2})
        check(len(doc["lines"][li2]["words"]) == len(doc["lines"][li2]["tokens"]) + 1, "split adds a box")
        check(doc["lines"][li2]["words"][-1]["text"] is None, "the surplus box has no word")
        try:
            store.act("synthbook", 1, li2, "accept_line", {})
            check(False, "accept refused while box count != word count")
        except ba.AuditError:
            check(True, "accept refused while box count != word count")
        doc = store.act("synthbook", 1, li2, "merge", {"word": 2})
        check(doc["lines"][li2]["count_ok"], "merge restores the count")
        doc, rec = store.undo("synthbook", 1)
        check(rec["op"] == "merge" and not doc["lines"][li2]["count_ok"], "undo reverts the merge")
        doc = store.act("synthbook", 1, li2, "merge", {"word": 2})
        rev = doc["revision"]
        try:
            store.act("synthbook", 1, li2, "accept_line", {}, rev=rev - 1)
            check(False, "a stale revision is refused")
        except ba.Conflict:
            check(True, "a stale revision is refused")

        # save / resume: a fresh store (server restart) sees the same state
        store2 = ba.AuditStore(root)
        doc2 = store2.get("synthbook", 1)
        check(doc2["revision"] == rev and doc2["lines"][li2]["state"] == "editing"
              and doc2["lines"][li]["state"] == "accepted", "state survives a restart")
        doc2, rec = store2.undo("synthbook", 1)
        check(rec["op"] == "merge", "undo works after a restart")
        doc2 = store2.act("synthbook", 1, li2, "merge", {"word": 2})

        # text flag allows accepting a line whose words cannot match
        li3 = next(i for i, l in enumerate(doc2["lines"]) if len(l["words"]) >= 3 and i not in (li, li2))
        doc2 = store2.act("synthbook", 1, li3, "delete_box", {"word": 1})
        try:
            store2.act("synthbook", 1, li3, "accept_line", {})
            check(False, "accept refused with a missing box")
        except ba.AuditError:
            check(True, "accept refused with a missing box")
        try:
            store2.act("synthbook", 1, li3, "flag_text", {"note": ""})
            check(False, "a text flag needs a note")
        except ba.AuditError:
            check(True, "a text flag needs a note")
        doc2 = store2.act("synthbook", 1, li3, "flag_text", {"note": "a word is missing from the text"})
        doc2 = store2.act("synthbook", 1, li3, "accept_line", {})
        check(doc2["lines"][li3]["state"] == "fixed" and doc2["lines"][li3]["text_flag"]["note"],
              "flagged line can be accepted with a box missing")
        doc2 = store2.act("synthbook", 1, li3, "add_box", {"box": [0, -5, 10, 5], "index": 1})
        doc2 = store2.act("synthbook", 1, li3, "unflag_text", {})

        # line edits
        before = copy.deepcopy(doc2["lines"][li]["frame"])
        doc2 = store2.act("synthbook", 1, li, "rotate_line", {"delta_deg": 0.5})
        doc2 = store2.act("synthbook", 1, li, "move_line", {"du": 0, "dv": 3})
        check(abs(doc2["lines"][li]["frame"]["angle_deg"] - before["angle_deg"] - 0.5) < 1e-6
              and doc2["lines"][li]["state"] == "editing", "rotate and move a line (re-opens it)")
        doc2 = store2.act("synthbook", 1, li, "resegment", {})
        check(len(doc2["lines"][li]["words"]) == len(doc2["lines"][li]["tokens"]), "re-split words inside a line")
        if n >= 2:
            t_before = (len(doc2["lines"][0]["tokens"]), len(doc2["lines"][1]["tokens"]))
            if t_before[0] >= 2:
                doc2 = store2.act("synthbook", 1, 0, "push_word", {})
                doc2 = store2.act("synthbook", 1, 0, "pull_word", {})
                check((len(doc2["lines"][0]["tokens"]), len(doc2["lines"][1]["tokens"])) == t_before,
                      "push and pull a word across a line break")

        # ---------------------------------------------------------- moving a word across a line break
        def inside(line):
            lb = line["box"]
            return all(lb[0] - 0.01 <= w["box"][0] and w["box"][2] <= lb[2] + 0.01
                       and lb[1] - 0.01 <= w["box"][1] and w["box"][3] <= lb[3] + 0.01 for w in line["words"])

        pi = next(i for i in range(n - 1) if len(doc2["lines"][i]["tokens"]) >= 3 and len(doc2["lines"][i + 1]["tokens"]) >= 3)
        for kind in ("push_word", "pull_word"):
            before = copy.deepcopy(doc2["lines"][pi:pi + 2])
            doc2 = store2.act("synthbook", 1, pi, kind, {})
            A, B = doc2["lines"][pi], doc2["lines"][pi + 1]
            check(A["count_ok"] and B["count_ok"], f"{kind}: both lines have one box per word")
            check(inside(A) and inside(B), f"{kind}: every box lies inside its own line's outline")
            check(A["box"] == before[0]["box"] and B["box"] == before[1]["box"], f"{kind}: neither outline changes")
            give_before, give_after = (before[0], A) if kind == "push_word" else (before[1], B)
            kept = [w["box"] for w in give_after["words"]]
            exp = [w["box"] for w in give_before["words"]]
            exp = exp[:-1] if kind == "push_word" else exp[1:]
            check(kept == exp, f"{kind}: the giving line keeps its other boxes and drops only the moved word's")
            take = B if kind == "push_word" else A
            check(all(w["state"] == "draft" for w in take["words"]) and all(
                "re-split after a word moved across the line break" in w["reasons"] for w in take["words"]),
                f"{kind}: the receiving line is re-split and marked uncertain")
            doc2, rec = store2.undo("synthbook", 1)
            check(rec["op"] == kind and doc2["lines"][pi:pi + 2] == before, f"{kind}: one undo restores both lines")
        # a box centred on another printed line is refused (outline cannot be stretched over a neighbour)
        L0 = doc2["lines"][pi]
        far = list(L0["words"][0]["box"])
        h = L0["box"][3] - L0["box"][1]
        far[1] -= 2 * h; far[3] -= 2 * h
        try:
            store2.act("synthbook", 1, pi, "set_word_box", {"word": 0, "box": far})
            check(False, "a box on another printed line is refused")
        except ba.AuditError:
            check(True, "a box on another printed line is refused")
        # R recovers a line whose outline was inflated by a stray box (state left by the old push_word)
        raw = json.loads(ba.audit_path(root, "synthbook", 1).read_text(encoding="utf-8"))
        prev, Lx = raw["lines"][pi], raw["lines"][pi + 1]
        orig_h = Lx["box"][3] - Lx["box"][1]
        stray = ba.quad_to_frame_box(Lx["frame"], ba.box_quad_px(prev["frame"], prev["words"][-1]["box"]))
        Lx["tokens"].insert(0, prev["tokens"].pop())
        prev["words"].pop()
        Lx["words"].insert(0, dict(box=ba.fix_box(stray), state="fixed", uncertain=False, reasons=[]))
        ba.save_audit(root, raw)          # derive() stretches the outline over the stray box, as before the fix
        st3 = ba.AuditStore(root)
        infl = st3.get("synthbook", 1)["lines"][pi + 1]
        check(infl["box"][3] - infl["box"][1] > 1.5 * orig_h, "fixture: stray box inflated the outline")
        doc3 = st3.act("synthbook", 1, pi + 1, "resegment", {})
        R = doc3["lines"][pi + 1]
        check(R["box"][3] - R["box"][1] <= orig_h + 8 and R["count_ok"] and inside(R),
              "R shrinks an inflated outline back and re-splits inside it")
        doc2 = st3.get("synthbook", 1)
        store2 = st3

        # ---------------------------------------------------------- page audited rule
        for i in range(n):
            if doc2["lines"][i]["state"] not in ba.DONE:
                doc2 = store2.act("synthbook", 1, i, "accept_line", {})
            if i < n - 1:
                check(doc2["status"] != "audited", "not audited while a line is unreviewed") if i == n - 2 else None
        check(doc2["status"] == "audited" and doc2["audited_at"], "page audited once every line is accepted")
        check(ba.validate_page(doc2) == [], "audited page passes the data contract")
        states = {w["state"] for l in doc2["lines"] for w in l["words"]}
        check(states <= {"accepted", "fixed"} and "fixed" in states, "words are accepted or fixed")
        wf_errs = wf.validate_lines(doc2["page_id"], ba.to_workflow_lines(doc2))
        check(wf_errs == [], "audited page exports to the annotation-workflow line shape")
        doc2 = store2.act("synthbook", 1, 0, "set_word_box", {"word": 0, "box": doc2["lines"][0]["words"][0]["box"]})
        check(doc2["status"] == "in_progress" and doc2["audited_at"] is None, "an edit re-opens an audited page")
        doc2, _ = store2.undo("synthbook", 1)
        check(doc2["status"] == "audited", "undo of that edit restores the audited state")
        check(ba.file_sha(out) == draft_sha, "the untouched draft file never changes")

        # ---------------------------------------------------------- pilot import (synthetic fixture)
        pdf_sha = ba.file_sha(root / "books" / "synthbook" / "source.pdf")
        pid = ba.page_id_for("synthbook", 2, pdf_sha)
        # second page: same image, second PDF page
        d = fitz.open(str(root / "books" / "synthbook" / "source.pdf"))
        d.insert_pdf(fitz.open(str(root / "books" / "synthbook" / "source.pdf")))
        d.save(str(root / "books" / "synthbook" / "source2.pdf"))
        (root / "books" / "synthbook" / "source2.pdf").replace(root / "books" / "synthbook" / "source.pdf")
        pdf_sha = ba.file_sha(root / "books" / "synthbook" / "source.pdf")
        pid = ba.page_id_for("synthbook", 2, pdf_sha)
        import shutil
        shutil.copy(root / "books" / "synthbook" / "pages" / "hires" / "0001.png",
                    root / "books" / "synthbook" / "pages" / "hires" / "0002.png")
        fx = dict(source=dict(slug="synthbook", page=2, page_id=pid, image_path="x.png"),
                  audit=dict(auditor="test", audited_at="2026-01-01T00:00:00+00:00", decision="approved"),
                  page=dict(lines=[dict(line_id=f"{pid}:l0", order=0, rect=[0.1, 0.1, 0.9, 0.13], words=[
                      dict(word_id=f"{pid}:l0:w0", order=0, text="الف", rect=[0.6, 0.105, 0.9, 0.125]),
                      dict(word_id=f"{pid}:l0:w1", order=1, text="ب", rect=[0.1, 0.105, 0.5, 0.125])])],
                            regions=[]))
        fxp = root / "fixture.json"
        fxp.write_text(json.dumps(fx, ensure_ascii=False), encoding="utf-8")
        pp = ba.import_pilot(root, fixture_path=fxp)
        pdoc = json.loads(pp.read_text(encoding="utf-8"))
        check(ba.validate_page(pdoc) == [] and pdoc["status"] == "previously_audited", "audited fixture imports")
        check(abs(pdoc["lines"][0]["words"][1]["rect"][0] - 0.1) < 1e-4, "fixture rectangles kept")

        # ---------------------------------------------------------- HTTP tool
        port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", port), ba.make_handler(ba.AuditStore(root), port))
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        base = f"http://127.0.0.1:{port}"
        try:
            st, html = http(base, "/")
            html = html.decode("utf-8")
            check(st == 200 and "box-audit server is not running" in html and "--port " in html,
                  "page explains a dead server and how to restart it")
            st, d = http(base, "/api/pages")
            check(st == 200 and {p["key"] for p in d["pages"]} == {"synthbook/1", "synthbook/2"}, "pages listed")
            st, d = http(base, "/api/page?key=synthbook/1")
            rev = d["doc"]["revision"]
            st, d = http(base, "/api/op", dict(key="synthbook/1", line=0, op="accept_line", args={}, rev=rev))
            check(st == 200 and d["ok"] and d["doc"]["revision"] == rev + 1, "op over HTTP saves and answers")
            st, d = http(base, "/api/op", dict(key="synthbook/1", line=0, op="nope", args={}, rev=rev + 1))
            check(st == 400 and "unknown action" in d["error"], "a bad action gets a readable error")
            st, d = http(base, "/api/op", dict(key="synthbook/1", line=0, op="accept_line", args={}, rev=0))
            check(st == 409 and "doc" in d, "stale revision answered with the saved page")
            st, d = http(base, "/api/undo", dict(key="synthbook/1"))
            check(st == 200 and d["ok"], "undo over HTTP")
            st, img = http(base, "/img?key=synthbook/1")
            check(st == 200 and img[:4] == b"\x89PNG", "page image served")
        finally:
            httpd.shutdown()
            httpd.server_close()

    # ---------------------------------------------------------- review.py message
    src = (ROOT / "farsi2epub" / "review.py").read_text(encoding="utf-8")
    check("the review server is not running" in src, "review feedback buttons explain a dead server")

    print(f"\n{'FAILED' if FAILS else 'PASSED'}: {len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
