#!/usr/bin/env python3
"""Regression test for tests/text_audit.py. Synthetic strings only; no books/ needed.

    ./venv/bin/python tests/text_audit_regression.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import text_audit as T  # noqa: E402
import text_audit_reflow as R  # noqa: E402

ZW = "‌"
DRAFT = (
    "کتاب ⟦خوب|خوب تر|∅؟⟧ است\n"                # 1: alternatives incl. nothing
    "⟦↵؟⟧ باغ گل\n"                             # 2,3: a disputed break
    "⟦↵؟⟧ سرو سیب\n"
    f"میز ⟦ها|{ZW}ها؟⟧ بزرگ\n"                  # 4: ZWNJ difference
    "⟦خانه کوچک|∅؟⟧\n"                          # 5: whole line seen by one reader
    "آخر ⟦؟⟧ کار ⟦ب …|بد|…؟⟧\n"                 # 6: unreadable + gap marks
    "پایان ⟦یک دو|یکدو؟⟧\n"                     # 7: join versus split
)


def geometry(root, page):
    crops = []
    for k in range(4):
        crops.append(dict(id=f"c{k + 1:02d}", order=k, file=None, kind="L", auto_line=k + 1 if k < 3 else None,
                          xs=[0.0, 100.0], ys=[10.0 * k, 10.0 * k], x0=0.0, x1=100.0, up=5.0, dn=5.0,
                          strip=dict(w=150, h=60, band_top=20, band_bottom=40, margin=24, up_p=8.5, dn_p=8.5, ext=20)))
    return crops, dict(file="x.png", width=100, height=100, kind="synthetic", sha256="0"), {1: "c01", 2: "c02", 3: "c03"}


def fresh(draft: str = DRAFT):
    root = Path(tempfile.mkdtemp(prefix="text_audit_reg_"))
    t = T.truth_dir(root)
    t.mkdir(parents=True)
    (t / "0003.lines.txt").write_bytes(draft.encode("utf-8"))
    return root


def store(root):
    return T.Store(root, geometry=geometry, log=lambda *a: None, consolidate=False)   # the draft's own lines


def act(s, op, rev=None, **args):
    d = s.get(3)
    return s.act(3, op, args, d["revision"] if rev is None else rev)


def raises(fn, exc=T.AuditError, needle=None):
    try:
        fn()
    except exc as e:
        if needle and needle not in str(e):
            raise AssertionError(f"wrong message: {e}")
        return
    raise AssertionError("expected " + exc.__name__)


def export(root) -> str:
    return (T.truth_dir(root) / "0003.lines.txt").read_bytes().decode("utf-8")


def mark_ids(doc, li):
    return [s["id"] for s in doc["lines"][li]["segs"] if s["t"] == "mark"]


def test_parse():
    ids = T.Ids()
    segs = T.parse_text("a ⟦x|y؟⟧ b ⟦؟⟧ ⟦∅|…؟⟧ ⟦↵؟⟧", ids)
    kinds = [s["t"] for s in segs]
    assert kinds == ["text", "mark", "text", "mark", "text", "mark", "text", "brk"], kinds
    assert segs[1]["alts"] == ["x", "y"] and segs[3]["alts"] == [] and segs[5]["alts"] == ["∅", "…"]
    assert all(s.get("choice") is None for s in segs if s["t"] == "mark")
    for bad in ("⟦x|y⟧", "a ⟦ b", "a ⟧", "⟦a⟦b؟⟧؟⟧"):
        raises(lambda b=bad: T.parse_text(b, T.Ids()))
    typed = T.parse_text("a ⟦؟⟧ b", T.Ids(), resolve_unreadable=True)
    assert typed[1]["choice"] == {"k": "unreadable"}


def test_roundtrip():
    root = fresh()
    s = store(root)
    d = s.get(3)
    assert export(root) == DRAFT                                   # untouched export is the draft
    assert T.export_text(s.open_page(3)) == DRAFT
    # every marker to its first alternative: expected string by hand
    for li in range(len(d["lines"])):
        for m in mark_ids(s.get(3), li):
            doc = s.get(3)
            seg = next(x for x in doc["lines"][li]["segs"] if x.get("id") == m)
            if seg["alts"]:
                s.act(3, "choose", dict(line=li, mark=m, choice=dict(k="alt", i=0)), doc["revision"])
    want = ("کتاب خوب است\n⟦↵؟⟧ باغ گل\n⟦↵؟⟧ سرو سیب\nمیز ها بزرگ\nخانه کوچک\n"
            "آخر ⟦؟⟧ کار ب …\nپایان یک دو\n")
    assert export(root) == want, export(root)


def test_export_formatting():
    ids = T.Ids()
    def r(text, *choices):
        segs = T.parse_text(text, ids)
        k = 0
        for sg in segs:
            if sg["t"] == "mark":
                sg["choice"] = choices[k]
                k += 1
        return T.render_segs(segs)
    E = dict(k="empty")
    assert r("a ⟦x|∅؟⟧ b", E) == "a b"                              # middle
    assert r("⟦x|∅؟⟧ a", E) == "a"                                   # start
    assert r("a ⟦x|∅؟⟧", E) == "a"                                   # end
    assert r("a⟦x|∅؟⟧ b", E) == "a b"                                # glued left: no space lost
    assert r("a ⟦x|∅؟⟧b", E) == "a b"
    assert r("a ⟦x|∅؟⟧ ⟦y|∅؟⟧ b", E, E) == "a b"
    assert r("⟦x|∅؟⟧", E) == ""
    assert r("a ⟦x|y؟⟧ b", dict(k="alt", i=1)) == "a y b"
    assert r("a ⟦x|y؟⟧ b", dict(k="custom", v="")) == "a b"
    assert r("a ⟦x|y؟⟧ b", dict(k="custom", v=f"گ{ZW}ل  م")) == f"a گ{ZW}ل  م b"   # nothing normalised
    assert r("a ⟦x|…؟⟧ b", dict(k="alt", i=1)) == "a ⟦؟⟧ b"
    assert r("a ⟦x|∅؟⟧ b", dict(k="alt", i=1)) == "a b"
    assert r("⟦یک دو|یکدو|یک‌دو؟⟧", dict(k="alt", i=1)) == "یکدو"
    assert r("⟦یک دو|یکدو|یک‌دو؟⟧", dict(k="alt", i=2)) == "یک‌دو"
    assert r("⟦یک دو|یکدو؟⟧", dict(k="alt", i=0)) == "یک دو"
    assert "differ" in T.invisible_note(["a b", "ab", "c"]) and T.invisible_note(["a", "b"]) == ""


def test_draft_copy():
    root = fresh()
    s = store(root)
    s.get(3)
    P = T.paths_for(root, 3)
    assert P["draft"].read_bytes() == DRAFT.encode()
    doc = s.get(3)
    act(s, "choose", line=0, mark=mark_ids(doc, 0)[0], choice=dict(k="empty"))
    assert P["draft"].read_bytes() == DRAFT.encode()                 # never rewritten
    assert export(root) != DRAFT
    P["draft"].write_bytes(b"changed")
    raises(lambda: store(root).open_page(3), needle="does not match")
    P["draft"].write_bytes(DRAFT.encode())
    # an export without a state file is never overwritten
    root2 = fresh()
    P2 = T.paths_for(root2, 3)
    P2["draft"].write_bytes(DRAFT.encode())
    P2["lines"].write_bytes(b"something else")
    raises(lambda: store(root2).open_page(3), needle="no state")
    # a state whose export was edited outside the tool: refuse to overwrite
    s3 = store(root)
    P["lines"].write_bytes(b"edited by hand")
    s3b = store(root)
    raises(lambda: s3b.open_page(3), needle="outside")


def test_line_ops_and_undo():
    root = fresh()
    s = store(root)
    n0 = len(s.get(3)["lines"])
    base = export(root)
    # split at a word (line 0: "کتاب ⟦...⟧ است" split before the marker)
    d = s.get(3)
    act(s, "split", line=0, seg=1, offset=0)
    d = s.get(3)
    assert len(d["lines"]) == n0 + 1 and d["lines"][0]["out"] == "کتاب" and d["lines"][1]["crop"] is None
    # join
    act(s, "join", line=0)
    d = s.get(3)
    assert len(d["lines"]) == n0 and d["lines"][0]["out"].startswith("کتاب ⟦")
    # insert / delete
    act(s, "insert", at=2)
    d = s.get(3)
    assert len(d["lines"]) == n0 + 1 and d["lines"][2]["out"] == ""
    act(s, "delete", line=2)
    assert len(s.get(3)["lines"]) == n0
    act(s, "delete", line=4)
    assert len(s.get(3)["lines"]) == n0 - 1
    # edit an unmarked word (the last text of line 0), then edit to nothing
    d = s.get(3)
    seg = d["lines"][0]["segs"][-1]
    act(s, "edit", line=0, seg=len(d["lines"][0]["segs"]) - 1, start=1, end=1 + len("است"), text="هست")
    assert s.get(3)["lines"][0]["out"].endswith("هست")
    act(s, "edit", line=0, seg=len(s.get(3)["lines"][0]["segs"]) - 1, start=1, end=1 + len("هست"), text="")
    assert s.get(3)["lines"][0]["out"].endswith("⟧")
    # set_text with a typed unreadable marker
    act(s, "set_text", line=0, text="سلام ⟦؟⟧ دنیا")
    d = s.get(3)
    assert d["lines"][0]["out"] == "سلام ⟦؟⟧ دنیا" and d["lines"][0]["unres"] == 0
    # crops
    act(s, "shift_crops", line=0, delta=1)
    d = s.get(3)
    assert [l["crop"] for l in d["lines"][:4]] == ["c02", "c03", "c04", None] or d["lines"][0]["crop"] == "c02"
    act(s, "set_crop", line=0, crop="c04")
    assert s.get(3)["lines"][0]["crop"] == "c04"
    act(s, "set_crop", line=0, crop=None)
    assert s.get(3)["lines"][0]["crop"] is None
    raises(lambda: act(s, "set_crop", line=0, crop="zz"))
    # undo everything, one step at a time; the export returns to the draft
    steps = 0
    while s.can_undo(3):
        s.undo(3, s.get(3)["revision"])
        steps += 1
    assert steps == 11, steps
    assert export(root) == base == DRAFT
    assert T.export_text(s.open_page(3)) == DRAFT
    raises(lambda: s.undo(3, s.get(3)["revision"]), needle="nothing to undo")
    # the log is append-only: ops + undos
    recs = [json.loads(x) for x in T.paths_for(root, 3)["undo"].read_text().splitlines()]
    assert len(recs) == 22 and [r["seq"] for r in recs] == list(range(1, 23))


def test_break():
    for which in ("two", "one"):
        root = fresh()
        s = store(root)
        d = s.get(3)
        assert d["pairs"]["1"] == [1, 2]
        act(s, "brk", line=1, which=which)
        d = s.get(3)
        if which == "two":
            assert [l["out"] for l in d["lines"][1:3]] == ["باغ گل", "سرو سیب"] and len(d["lines"]) == 7
        else:
            assert d["lines"][1]["out"] == "باغ گل سرو سیب" and len(d["lines"]) == 6
        assert "⟦↵؟⟧" not in export(root)
        s.undo(3, d["revision"])
        assert export(root) == DRAFT
    # the partner line works too, and a lone marker is just removed
    root = fresh()
    s = store(root)
    act(s, "brk", line=2, which="one")
    assert s.get(3)["lines"][1]["out"] == "باغ گل سرو سیب"
    root = fresh("⟦↵؟⟧ تنها\nدیگر\n")
    s = store(root)
    act(s, "brk", line=0, which="one")
    assert [l["out"] for l in s.get(3)["lines"]] == ["تنها", "دیگر"]


def test_accept_and_audited():
    root = fresh("یک ⟦الف|ب؟⟧\nدو\n\nسه\n")
    s = store(root)
    d = s.get(3)
    assert [l["state"] for l in d["lines"]] == ["open", "open", "accepted", "open"]   # blank line needs no reading
    raises(lambda: act(s, "accept", line=0), needle="unresolved")
    act(s, "accept", line=1)                                          # a line without markers still needs accept
    assert s.get(3)["lines"][1]["state"] == "accepted"
    act(s, "choose", line=0, mark=mark_ids(s.get(3), 0)[0], choice=dict(k="alt", i=1))
    assert s.get(3)["lines"][0]["state"] == "open"
    act(s, "accept", line=0)
    assert not s.get(3)["summary"]["audited"]
    state = json.loads(T.paths_for(root, 3)["state"].read_bytes())
    assert state["status"] == "in_progress" and state["export"]["final"] is False
    act(s, "accept", line=3)
    d = s.get(3)
    assert d["summary"]["audited"] and d["status"] == "audited" and d["export"]["final"] is True
    assert export(root) == "یک ب\nدو\n\nسه\n"
    # any later edit re-opens the line and the page is no longer audited
    act(s, "edit", line=1, seg=0, start=0, end=2, text="دوم")
    d = s.get(3)
    assert d["lines"][1]["state"] == "open" and d["status"] == "in_progress" and not d["export"]["final"]
    act(s, "accept", line=1)
    assert s.get(3)["status"] == "audited"
    act(s, "reopen", line=1)
    assert s.get(3)["status"] == "in_progress"
    # gap marks left in a line need an explicit confirmation
    act(s, "set_text", line=1, text="دو … سه")
    raises(lambda: act(s, "accept", line=1), needle="NEEDS_CONFIRM")
    act(s, "accept", line=1, allow_gap_marks=True)


def test_resume_and_stale():
    root = fresh()
    s = store(root)
    d = s.get(3)
    rev0 = d["revision"]
    act(s, "choose", line=0, mark=mark_ids(d, 0)[0], choice=dict(k="alt", i=1))
    act(s, "cursor", rev=None, line=3, mark=None)
    s2 = store(root)                                                  # "restart"
    d2 = s2.get(3)
    assert d2["revision"] == rev0 + 1 and d2["cursor"]["line"] == 3
    assert d2["lines"][0]["out"].startswith("کتاب خوب تر")
    assert s2.can_undo(3) == 1
    s2.undo(3, d2["revision"])
    assert export(root) == DRAFT
    # stale revision: a second tab holding the old revision must not overwrite
    s3 = store(root)
    cur = s3.get(3)["revision"]
    s3.act(3, "choose", dict(line=0, mark=mark_ids(s3.get(3), 0)[0], choice=dict(k="alt", i=0)), cur)
    raises(lambda: s3.act(3, "accept", dict(line=1), cur), exc=T.Conflict)
    raises(lambda: s3.undo(3, cur), exc=T.Conflict)
    # stale line identity
    doc = s3.get(3)
    raises(lambda: s3.act(3, "accept", dict(line=1, lid=999), doc["revision"]), exc=T.Conflict)
    # a lost last log record (crash after state write is impossible; before it is healed)
    P = T.paths_for(root, 3)
    lines = P["undo"].read_bytes().splitlines(keepends=True)
    state_before = P["state"].read_bytes()
    s4 = store(root)
    d4 = s4.get(3)
    s4.act(3, "choose", dict(line=3, mark=mark_ids(d4, 3)[0], choice=dict(k="alt", i=0)), d4["revision"])
    P["state"].write_bytes(state_before)                              # simulate: log written, state not
    s5 = store(root)
    assert s5.get(3)["lines"][3]["out"].startswith("میز ها") and export(root).splitlines()[3].startswith("میز ها")


def test_page_refusal():
    root = fresh()
    s = store(root)
    for p in (1, 2, 6, 0):
        raises(lambda p=p: s.open_page(p), needle="not one of the pages")
    raises(lambda: T.paths_for(root, 2) and T.check_page(2))
    t = T.truth_dir(root)
    assert sorted(x.name for x in t.iterdir()) == ["0003.lines.txt"]    # nothing written for the refused pages
    s.get(3)
    names = sorted(x.name for x in t.iterdir())
    assert names == ["0003.lines.draft.txt", "0003.lines.txt", "0003.text_audit.json"], names


def test_http():
    root = fresh()
    s = store(root)
    h = T.make_handler(s, 0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), h)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"

    def req(path, body=None):
        r = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r) as x:
                return x.status, json.loads(x.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert req("/api/ping")[1]["ok"]
        code, d = req("/api/page?page=3")
        rev = d["doc"]["revision"]
        mid = mark_ids(d["doc"], 0)[0]
        code, r = req("/api/op", dict(page=3, op="choose", rev=rev, args=dict(line=0, mark=mid, choice=dict(k="alt", i=0))))
        assert code == 200 and r["doc"]["revision"] == rev + 1
        code, r = req("/api/op", dict(page=3, op="accept", rev=rev, args=dict(line=1)))
        assert code == 409 and r["doc"]["revision"] == rev + 1                # stale tab
        code, r = req("/api/op", dict(page=2, op="accept", rev=0, args=dict(line=0)))
        assert code == 400 and "not one of the pages" in r["error"]
        code, r = req("/api/op", dict(page=3, op="nonsense", rev=rev + 1, args={}))
        assert code == 400
        code, r = req("/api/undo", dict(page=3, rev=rev + 1))
        assert code == 200 and r["undone"] == "choose"
        code, r = req("/api/page?page=9")
        assert code == 400
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "not running" in html and "text_audit.py serve" in html
    finally:
        httpd.shutdown()
        httpd.server_close()


def synth_crops(widths, ys, heights=None):
    out = []
    for k, (w, y) in enumerate(zip(widths, ys)):
        h = (heights or [50] * len(widths))[k]
        out.append(dict(id=f"c{k + 1:02d}", order=k, file=None, kind="X", xs=[0.0, float(w)], ys=[float(y), float(y)],
                        x0=0.0, x1=float(w), up=h / 2, dn=h / 2, ink_w=float(w), mass=1000,
                        strip=dict(w=w + 48, h=60, band_top=20, band_bottom=40, margin=24, up_p=8.5, dn_p=8.5, ext=20)))
    return out


def test_auto_assign():
    # a short, isolated strip at the top (the page number) that the draft has no line for
    draft = "ابجد هوز حطی کلمن سعفص قرشت ثخذ ضظغ\nبسیار طولانی ترین سطر این صفحه برای آزمایش است و بس\nسطر کوتاه\nآخرین سطر این صفحه تمام\n"
    exp = [T.expected_chars(T.parse_text(l, T.Ids())) for l in draft.splitlines()]
    widths = [90] + [exp[0] * 15, exp[1] * 15, exp[2] * 15, exp[3] * 15] + [400, 300]
    ys = [20, 200, 280, 360, 440, 600, 690]
    crops = synth_crops(widths, ys, [30, 50, 50, 50, 50, 120, 110])
    got = T.auto_assign(crops, exp, [T.plain_first(T.parse_text(l, T.Ids())) for l in draft.splitlines()])
    assert got == {1: "c02", 2: "c03", 3: "c04", 4: "c05"}, got              # page-number strip and trailing strips unused
    # a page-number line in the draft does take the page-number strip
    d2 = "-۳-\n" + draft
    exp2 = [T.expected_chars(T.parse_text(l, T.Ids())) for l in d2.splitlines()]
    got2 = T.auto_assign(crops, exp2, [T.plain_first(T.parse_text(l, T.Ids())) for l in d2.splitlines()])
    assert got2[1] == "c01" and got2[2] == "c02", got2
    # an underline-thin strip next to a heading is never used for a text line
    crops3 = synth_crops([300, 280, 600], [100, 140, 220], [50, 20, 50])
    got3 = T.auto_assign(crops3, [8, 40], ["عنوان", "x" * 40])
    assert got3 == {1: "c01", 2: "c03"}, got3


def geometry_synth(root, page):
    crops = synth_crops([90, 300, 450, 600], [20, 200, 280, 360], [30, 50, 50, 50])
    return crops, dict(file="x.png", width=100, height=100, kind="synthetic", sha256="0"), None


def test_unmatched_strips_and_crop_actions():
    draft = "عنوان کوتاه سطر\nسطر بلندتر دوم این صفحه\nآخر\n"
    root = Path(tempfile.mkdtemp(prefix="text_audit_reg_"))
    t = T.truth_dir(root)
    t.mkdir(parents=True)
    (t / "0003.lines.txt").write_bytes(draft.encode())
    s = T.Store(root, geometry=geometry_synth, log=lambda *a: None)
    d = s.get(3)
    used = [l["crop"] for l in d["lines"]]
    assert "c01" not in used, used                                     # the short top strip is left for the owner to see
    assert d["ignored"] == []
    base = export(root)
    # add a line for the unmatched strip, in place
    d = s.get(3)
    s.act(3, "add_line_for_crop", dict(at=0, crop="c01"), d["revision"])
    d = s.get(3)
    assert len(d["lines"]) == 4 and d["lines"][0]["crop"] == "c01" and d["lines"][0]["out"] == "" and d["lines"][0]["crop_src"] == "owner"
    s.undo(3, d["revision"])
    assert export(root) == base and len(s.get(3)["lines"]) == 3
    # ignore it, restore it, undo both
    d = s.get(3)
    before = json.dumps(d["lines"], sort_keys=True)
    s.act(3, "ignore_crop", dict(crop="c01", ignore=True), d["revision"])
    assert s.get(3)["ignored"] == ["c01"]
    d = s.get(3)
    s.act(3, "ignore_crop", dict(crop="c01", ignore=False), d["revision"])
    assert s.get(3)["ignored"] == []
    s.undo(3, s.get(3)["revision"])
    assert s.get(3)["ignored"] == ["c01"]
    s.undo(3, s.get(3)["revision"])
    assert s.get(3)["ignored"] == [] and json.dumps(s.get(3)["lines"], sort_keys=True) == before
    # ignoring then adding a line for the strip un-ignores it, and undo puts both back
    d = s.get(3)
    s.act(3, "ignore_crop", dict(crop="c01", ignore=True), d["revision"])
    d = s.get(3)
    s.act(3, "add_line_for_crop", dict(at=0, crop="c01"), d["revision"])
    assert s.get(3)["ignored"] == [] and len(s.get(3)["lines"]) == 4
    s.undo(3, s.get(3)["revision"])
    assert s.get(3)["ignored"] == ["c01"] and len(s.get(3)["lines"]) == 3
    s.undo(3, s.get(3)["revision"])
    # change one line's strip only, then line and all after it; undo restores exactly
    snap = json.dumps(s.get(3)["lines"], sort_keys=True)
    cur = [l["crop"] for l in s.get(3)["lines"]]
    assert cur == ["c02", "c03", "c04"], cur
    d = s.get(3)
    s.act(3, "set_crop", dict(line=1, crop="c04", following=False), d["revision"])
    assert [l["crop"] for l in s.get(3)["lines"]] == ["c02", "c04", "c04"]
    s.undo(3, s.get(3)["revision"])
    assert json.dumps(s.get(3)["lines"], sort_keys=True) == snap
    d = s.get(3)
    s.act(3, "set_crop", dict(line=0, crop="c01", following=True), d["revision"])     # everything moves up by one
    assert [l["crop"] for l in s.get(3)["lines"]] == ["c01", "c02", "c03"]
    raises(lambda: s.act(3, "set_crop", dict(line=0, crop="c01", following=True), s.get(3)["revision"]), needle="already")
    s.undo(3, s.get(3)["revision"])
    assert json.dumps(s.get(3)["lines"], sort_keys=True) == snap
    # a restart keeps the ignored list
    d = s.get(3)
    s.act(3, "ignore_crop", dict(crop="c01", ignore=True), d["revision"])
    s2 = T.Store(root, geometry=geometry_synth, log=lambda *a: None)
    assert s2.get(3)["ignored"] == ["c01"]
    raises(lambda: s2.act(3, "ignore_crop", dict(crop="zz"), s2.get(3)["revision"]))
    shutil.rmtree(root, ignore_errors=True)


# --- strip-first rebuild (text_audit_reflow) ---------------------------------

def _compose(tokens, mapping, k):
    """Text of the refined tokens when every small marker takes the piece of alternative k."""
    import copy
    toks = copy.deepcopy(tokens)
    pick = dict(mapping[k])
    for x in toks:
        if isinstance(x, dict):
            x["choice"] = {"k": "alt", "i": pick[x["id"]]}
    return T.render_segs(R.segs_of(["پیش"] + toks + ["پس"]))


def test_reflow_refine():
    ids = T.Ids(50, 1)
    cases = [
        ["تعیین میگرد ند", "تعیین میگردند اداره خواهد شد .", "نعیین میگردند"],
        ["مدت سه ماه پس از انقضاء سال", "مدت … ماه پس ازانقضا … سال"],
        ["شرکت (اصالتا یا وکالتا ) در مجمع حضور داشته باشند", "شرکت (اصالتاً یاوکالتاً) در مجمع حضورداشته باشند",
         "شرکت (اصالتاً یاوکالتاً) درمجمع حضورداشته باشند"],
        ["، اسناد ، سفته ها وچکها وتنزیل", "، اسناد ، سفته هاوچکهاوتنزیل", "،اسناد ،سفته ها وچکها ارتنزیل"],
        ["راد ارد ازجمله تغییر اساسنامه ،", "راداراست ازجمله تغییراساسنامه .", "راداردازجمله تغییراساسنامه ."],
    ]
    for alts in cases:
        m = {"t": "mark", "id": 7, "alts": sorted(alts), "choice": None}
        toks, mp, counts = R.refine(m, ids.mark)
        assert mp is not None, alts
        marks = [x for x in toks if isinstance(x, dict)]
        assert max(len(a) for x in marks for a in x["alts"]) < max(len(a) for a in alts), alts
        for k, a in enumerate(m["alts"]):                  # every reading can still be composed, exactly
            # (a lone "…" piece is offered as "could not be read", which exports as ⟦؟⟧)
            assert _compose(toks, mp, k).replace("⟦؟⟧", "…") == "پیش " + a + " پس", (a, _compose(toks, mp, k))
        assert len(counts) == len(toks) and all(counts[-1][k] == len(a.split(" ")) for k, a in enumerate(m["alts"]))
    toks, mp, _ = R.refine({"t": "mark", "id": 7, "alts": sorted(cases[0]), "choice": None}, ids.mark)
    got = [x if isinstance(x, str) else x["alts"] for x in toks]
    assert got == [["تعیین", "نعیین"], ["میگرد ند", "میگردند"], ["اداره خواهد شد .", "∅"]], got
    toks, _, _ = R.refine({"t": "mark", "id": 7, "alts": sorted(cases[1]), "choice": None}, ids.mark)
    assert [x for x in toks if isinstance(x, str)] == ["مدت", "ماه", "پس", "سال"], toks      # agreed words become plain text
    # left alone: one-word readings, a marker that may be absent, an unreadable reading, a marker already decided
    for alts, choice in ((["خوب", "خواب"], None), (["خوب تر", "خوب", "∅"], None), (["ب …", "بد", "…"], None),
                         (["یک دو", "یکدو سه"], {"k": "alt", "i": 0})):
        m = {"t": "mark", "id": 7, "alts": alts, "choice": choice}
        assert R.refine(m, ids.mark) == ([m], None, None)
    # tokens <-> segments round trip, and lines that are not plain space-separated tokens are refused
    segs = T.parse_text("ماده ⟦(۱۶)|۱۶)؟⟧ حد نصاب ⟦:|و؟⟧", T.Ids())
    assert T.render_segs(R.segs_of(R.tokens_of(segs))) == T.render_segs(segs)
    assert R.tokens_of(T.parse_text("دو  فاصله", T.Ids())) is None and R.tokens_of(T.parse_text("چسبیده⟦الف|ب؟⟧", T.Ids())) is None
    assert R.tokens_of(T.parse_text("⟦↵؟⟧ ", T.Ids()))[1:] == []


PX = 15
LONG1 = "این سطر نخست صفحه است و هیچ اختلافی در آن میان خوانندگان دیده نمیشود"
LONG4 = "این سطر پایانی صفحه است و هیچ اختلافی در آن میان خوانندگان دیده نمیشود"
A2, A3 = "هیئت مدیره از طرف مجمع صاحبان سهام تعیین میگردند و شرکت بوسیله آنان", "پس ازانقضاء دوره تصدی هیئت مدیره تشریفات مربوط صورت نگرفته است"
REFLOW_DRAFT = (
    LONG1 + "\n"
    "⟦↵؟⟧ " + A2 + " ⟦اداره خواهد شد .|∅؟⟧\n"                 # reader A ends the long line with the short one
    "⟦↵؟⟧ ⟦اداره خواهند شد . پس|پس؟⟧ " + A3[3:] + "\n"         # reader B starts the next line with it
    + LONG4 + "\n"
)


def geometry_reflow(root, page):
    widths = [len(LONG1) * PX, len(A2) * PX, len("اداره خواهد شد .") * PX, len(A3) * PX, len(LONG4) * PX]
    crops = synth_crops(widths, [100, 180, 260, 340, 420])
    return crops, dict(file="x.png", width=100, height=100, kind="synthetic", sha256="0"), None


def test_reflow_lines():
    root = fresh(REFLOW_DRAFT)
    s = T.Store(root, geometry=geometry_reflow, log=lambda *a: None)
    d = s.get(3)
    out = [l["out"] for l in d["lines"]]
    assert len(out) == 5 and [l["crop"] for l in d["lines"]] == ["c01", "c02", "c03", "c04", "c05"], out
    assert out[1] == A2 and out[3] == A3, out                    # the short printed line stands alone on its strip
    assert out[2] == "اداره ⟦خواهد|خواهند؟⟧ شد .", out[2]
    assert "↵" not in export(root) and "∅" not in export(root) and d["pairs"] == {}
    assert [l.get("reflowed", False) for l in d["lines"]] == [False, True, True, True, False]
    assert d["lines"][2]["src"] == [2, 3] and d["reflow"]["zones"] == [dict(lines=[2, 3], strips=3, result="3 lines")]
    assert (T.truth_dir(root) / "0003.lines.draft.txt").read_bytes().decode() == REFLOW_DRAFT   # the draft itself is untouched
    # a reader that broke one printed line into pieces is put back together on the one strip there is
    frag = (LONG1 + "\n⟦شرکت (اصالتا|∅؟⟧\n⟦یا وکالتا|∅؟⟧\n"
            "⟦) در مجمع حضور داشته باشند و تصمیمات این مجمع معتبر است|شرکت (اصالتاً یاوکالتاً) در مجمع حضورداشته باشند و تصمیمات این مجمع معتبر است|∅؟⟧\n"
            + LONG4 + "\n")

    def geo(root, page):
        crops = synth_crops([len(LONG1) * PX, 75 * PX, len(LONG4) * PX], [100, 180, 260])
        return crops, dict(file="x.png", width=100, height=100, kind="synthetic", sha256="0"), None
    root = fresh(frag)
    d = T.Store(root, geometry=geo, log=lambda *a: None).get(3)
    assert len(d["lines"]) == 3 and d["lines"][1]["crop"] == "c02", [l["out"] for l in d["lines"]]
    mid = d["lines"][1]
    assert "∅" not in mid["out"] and mid["out"].startswith("شرکت ⟦") and "و تصمیمات این مجمع معتبر است" in mid["out"], mid["out"]
    assert max(len(a) for sg in mid["segs"] if sg["t"] == "mark" for a in sg["alts"]) <= 20
    # no strip to go by: the draft's lines stay, with their disputed-break marks
    def geo0(root, page):
        return [], dict(file="x.png", width=100, height=100, kind="synthetic", sha256="0"), None
    root = fresh(DRAFT)
    d = T.Store(root, geometry=geo0, log=lambda *a: None).get(3)
    assert d["pairs"]["1"] == [1, 2] and len(d["lines"]) == 7


def test_migration():
    root = fresh(REFLOW_DRAFT)
    s = T.Store(root, geometry=geometry_reflow, log=lambda *a: None, consolidate=False)     # a version 1 state
    act(s, "accept", line=0)
    d = s.get(3)
    act(s, "choose", line=2, mark=mark_ids(d, 2)[0], choice={"k": "alt", "i": 0})
    act(s, "brk", line=1, which="one")
    P = T.paths_for(root, 3)
    old = json.loads(P["state"].read_bytes())
    old["tool_version"] = "text_audit/1"
    P["state"].write_bytes(T.canonical(old))
    old_state, old_undo = P["state"].read_bytes(), P["undo"].read_bytes()
    logs = []
    s2 = T.Store(root, geometry=geometry_reflow, log=logs.append)
    d = s2.get(3)
    t = T.truth_dir(root)
    assert (t / "0003.text_audit.v1.json").read_bytes() == old_state and (t / "0003.text_audit.undo.v1.jsonl").read_bytes() == old_undo
    assert not P["undo"].exists() and d["revision"] == 0 and d["tool_version"] == T.TOOL_VERSION
    assert [l["out"] for l in d["lines"]][1:4] == [A2, "اداره ⟦خواهد|خواهند؟⟧ شد .", A3]
    assert d["lines"][0]["state"] == "accepted" and d["migrated"]["lines_kept_accepted"] == 1
    assert d["migrated"]["choices_not_carried"] == [dict(old_line=2, choice="اداره خواهند شد . پس")], d["migrated"]
    assert export(root) == "\n".join(l["out"] for l in d["lines"]) + "\n" and logs
    # it happens once; the new state works and undoes like any other
    s3 = T.Store(root, geometry=geometry_reflow, log=lambda *a: None)
    assert s3.get(3)["migrated"]["at"] == d["migrated"]["at"]
    r = s3.act(3, "choose", dict(line=2, mark=mark_ids(d, 2)[0], choice={"k": "alt", "i": 1}), 0)
    assert r["lines"][2]["out"] == "اداره خواهند شد ."
    s3.undo(3, 1)
    assert s3.get(3)["lines"][2]["out"] == "اداره ⟦خواهد|خواهند؟⟧ شد ."
    # choices on markers that were only split word by word are carried, and the line stays accepted
    draft = LONG1 + "\nپایان ⟦یک دو سه|یک دوسه؟⟧ است\n" + LONG4 + "\n"

    def geo(root, page):
        crops = synth_crops([len(LONG1) * PX, 17 * PX, len(LONG4) * PX], [100, 180, 260])
        return crops, dict(file="x.png", width=100, height=100, kind="synthetic", sha256="0"), None
    root = fresh(draft)
    s = T.Store(root, geometry=geo, log=lambda *a: None, consolidate=False)
    act(s, "choose", line=1, mark=mark_ids(s.get(3), 1)[0], choice={"k": "alt", "i": 1})
    act(s, "accept", line=1)
    P = T.paths_for(root, 3)
    old = json.loads(P["state"].read_bytes())
    old["tool_version"] = "text_audit/1"
    P["state"].write_bytes(T.canonical(old))
    d = T.Store(root, geometry=geo, log=lambda *a: None).get(3)
    assert d["lines"][1]["out"] == "پایان یک دوسه است" and d["lines"][1]["state"] == "accepted", d["lines"][1]
    assert d["migrated"]["choices_carried"] == 1 and d["migrated"]["lines_kept_accepted"] == 1
    from text_audit_ui import PAGE_HTML
    assert "Edit the reading you chose for this spot" in PAGE_HTML and "s.alts[s.choice.i]" in PAGE_HTML


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    order = [test_parse, test_roundtrip, test_export_formatting, test_draft_copy, test_line_ops_and_undo,
             test_break, test_accept_and_audited, test_resume_and_stale, test_page_refusal, test_http,
             test_auto_assign, test_unmatched_strips_and_crop_actions,
             test_reflow_refine, test_reflow_lines, test_migration]
    assert set(order) == set(tests)
    made = []
    for t in order:
        try:
            t()
            print("ok  ", t.__name__)
        except Exception:
            import traceback
            traceback.print_exc()
            print("FAIL", t.__name__)
            return 1
    for p in Path(tempfile.gettempdir()).glob("text_audit_reg_*"):
        shutil.rmtree(p, ignore_errors=True)
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
