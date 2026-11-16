#!/usr/bin/env python3
"""Gemini box engine for review (``review --bbox-engine gemini``): synthetic, $0.

    ./venv/bin/python tests/gemini_boxes_regression.py

No network: ``hard_reader._gemini_post`` is replaced by a function that fails
the test, and every service gets a stub post function.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from click.testing import CliRunner  # noqa: E402

from farsi2epub import gemini_boxes as GB, hard_reader, review  # noqa: E402
from farsi2epub.cli import main  # noqa: E402
from farsi2epub.locate import Query  # noqa: E402


def _no_network(*_a, **_k):
    raise AssertionError("network call attempted")


hard_reader._gemini_post = _no_network

PNG = b"\x89PNG\r\n\x1a\n" + b"synthetic page image" * 20
MD = ("# سرفصل\n\n"
      "این یک متن **آزمایشی** است که در آن کتاب آمده و دوباره کتاب می‌آید تا\n"
      "سطر بعدی ادامه پیدا کند و پایان[^1] متن اینجاست.\n\n"
      "[^1]: یادداشت پایین صفحه")


def span_of(text: str, nth: int = 1) -> tuple[int, int]:
    i = -1
    for _ in range(nth):
        i = MD.index(text, i + 1)
    return i, i + len(text)


def make_ws(tmp: Path) -> SimpleNamespace:
    (tmp / "pages" / "hires").mkdir(parents=True)
    (tmp / "pages" / "hires" / "0001.png").write_bytes(PNG)
    (tmp / "source.pdf").write_bytes(b"%PDF-1.4 synthetic")
    return SimpleNamespace(root=tmp, slug="synthetic", pdf_path=tmp / "source.pdf",
                           page_hires_path=lambda n: tmp / "pages" / "hires" / f"{n:04d}.png",
                           page_image_path=lambda n: tmp / "pages" / f"{n:04d}.png",
                           review_dir=tmp / "review", meta={"page_count": 1})


def answer(passages: list, status: int = 200, usage=(1000, 200)):
    data = {"steps": [{"type": "model_output",
                       "content": [{"type": "text", "text": json.dumps({"passages": passages})}]}],
            "usage": {"total_input_tokens": usage[0], "total_output_tokens": usage[1]}}
    return status, data


class Stub:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, body, key):
        assert key == "fake-key"
        self.bodies.append(body)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def run() -> None:
    import box_options_eval as BOE
    tmp = Path(tempfile.mkdtemp(prefix="gemini_boxes_reg_"))
    try:
        ws = make_ws(tmp)

        # -- constants and request: one copy, eval and production identical -----
        check(BOE.A2_WORDING_W1 is GB.A2_WORDING_W1 and BOE.Answer1 is GB.Answer1
              and BOE.A2_MODEL == GB.A2_MODEL == "gemini-4-carbon", "eval imports the frozen constants from production")
        items = [{"before": "این یک متن", "text": "آزمایشی", "after": "است که در آن"},
                 {"before": "", "text": "سرفصل", "after": "این یک"}]
        for mo in (6000, 12000):
            check(json.dumps(GB.request_body(PNG, items, mo), sort_keys=True)
                  == json.dumps(BOE.a2_body("W1", PNG, items, mo), sort_keys=True),
                  f"request body equals tests/box_options_eval.a2_body (max_out {mo})")
        body = GB.request_body(PNG, items, 1000)
        check(body["generation_config"] == {"max_output_tokens": 1000, "thinking_level": "low"}
              and body["input"][1]["resolution"] == "ultra_high" and body["model"] == "gemini-4-carbon",
              "thinking low, ultra_high, gemini-4-carbon")
        check(GB.max_output_tokens(3) == 850 and GB.max_output_tokens(500) == 12000, "max output tokens sized and capped")

        # -- yxyx conversion and degenerate boxes ------------------------------------
        check(GB.box_to_segment([100, 200, 150, 600]) == {"x0": 0.2, "y0": 0.1, "x1": 0.6, "y1": 0.15},
              "box_2d [ymin,xmin,ymax,xmax]/1000 -> x0,y0,x1,y1")
        for bad in ([100, 200, 100, 600], [100, 200, 150, 200], [-1, 0, 10, 10], [0, 0, 10, 1001], [1, 2, 3], None):
            check(GB.box_to_segment(bad) is None, f"degenerate box rejected: {bad}")
        r = GB.placement_from_boxes([{"box_2d": [0, 0, 0, 0]}, {"box_2d": [500, 0, 2000, 10]}])
        check(r.status == "unresolved" and r.reason == "gemini: not found", "only bad boxes -> unresolved, not found")
        r = GB.placement_from_boxes([{"box_2d": [300, 100, 330, 400]}, {"box_2d": [200, 500, 230, 900]}])
        check(r.status == "located" and [s["y0"] for s in r.box["segments"]] == [0.2, 0.3]
              and r.box["source"] == "gemini" and abs(r.box["x0"] - 0.1) < 1e-9 and abs(r.box["x1"] - 0.9) < 1e-9,
              "wrapped boxes -> ordered per-line segments, union envelope, source gemini")

        # -- passage building ----------------------------------------------------------
        q_bold = Query("آزمایشی", span_of("آزمایشی"))
        p, _ = GB.passage_for(MD, q_bold)
        check(p == {"before": "سرفصل این یک متن", "text": "آزمایشی", "after": "است که در آن"},
              "passage: target and 4 context words each side, Markdown stripped")
        q_book2 = Query("کتاب", span_of("کتاب", 2))
        p2, _ = GB.passage_for(MD, q_book2)
        check(p2 == {"before": "کتاب آمده و دوباره", "text": "کتاب", "after": "می‌آید تا سطر بعدی"},
              "repeated word: the second occurrence gets its own context")
        q_wrap = Query("می‌آید تا\nسطر", span_of("می‌آید تا\nسطر"))
        p3, _ = GB.passage_for(MD, q_wrap)
        check(p3["text"] == "می‌آید تا سطر" and p3["after"] == "بعدی ادامه پیدا کند",
              "wrapped span: newline collapsed, context across the line break")
        q_fn = Query("پایان[^1]", span_of("پایان[^1]"))
        check(GB.passage_for(MD, q_fn)[0]["text"] == "پایان", "footnote mark stripped from target")
        q_ins = Query("", (span_of("دوباره")[1] + 1,) * 2, ("هم",), "insertion")
        p4, _ = GB.passage_for(MD, q_ins)
        check(p4["text"] == "دوباره کتاب" and p4["before"].endswith("و") and p4["after"].startswith("می‌آید"),
              "insertion: the words on either side are the target")
        q_none = Query("نیست در صفحه", None, kind="finding")
        check(GB.passage_for(MD, q_none) == (None, "gemini: correction has no span in the page text"),
              "no span in the page text -> reason")
        q_snip = Query("پیدا کند", None, kind="finding")
        check(GB.passage_for(MD, q_snip)[0]["text"] == "پیدا کند", "unlinked finding found once in the text gets a span")

        # -- service: one call per page, cache hit, results --------------------------
        queries = [q_bold, q_book2, q_wrap, q_none]
        stub = Stub([answer([{"id": 1, "boxes": [{"box_2d": [100, 300, 130, 500]}]},
                             {"id": 2, "boxes": []},
                             {"id": 3, "boxes": [{"box_2d": [140, 50, 170, 200]}, {"box_2d": [180, 700, 210, 950]}]},
                             {"id": 1, "boxes": [{"box_2d": [0, 0, 999, 999]}]}])])
        svc = GB.GeminiBoxService(ws, mode="auto", max_cost=5.0, api_key="fake-key", post=stub)
        res = svc.results(1, MD, queries)
        check([r.status for r in res] == ["pending", "pending", "pending", "unresolved"], "uncached page is pending")
        svc.refine(1, MD, queries)
        check(len(stub.bodies) == 1, "one request for all passages of the page")
        sent = stub.bodies[0]
        want = [GB.passage_for(MD, q)[0] for q in queries[:3]]
        check(json.dumps(sent, sort_keys=True) == json.dumps(BOE.a2_body("W1", PNG, want, GB.max_output_tokens(3)),
                                                            sort_keys=True), "sent body is the A2 request")
        res = svc.results(1, MD, queries)
        check([r.status for r in res] == ["located", "unresolved", "located", "unresolved"]
              and res[1].reason == "gemini: not found" and abs(res[0].box["x0"] - 0.3) < 1e-9
              and len(res[2].box["segments"]) == 2, "results: located / not found / wrapped / no span; duplicate id ignored")
        files = sorted((tmp / "locate_evidence" / "gemini_boxes").glob("*.json"))
        check(len([f for f in files if f.name != "last_run_cost.json"]) == 1, "page answer cached on disk")
        rc = json.loads((tmp / "locate_evidence" / "gemini_boxes" / "last_run_cost.json").read_text())
        check(rc["calls"] == 1 and abs(rc["spent"] - (1000 * GB.config.gemini_price()["in"] + 200 * GB.config.gemini_price()["out"]) / 1e6) < 1e-9,
              "receipt: calls and spent")
        svc.refine(1, MD, queries)
        svc.close()
        svc2 = GB.GeminiBoxService(ws, mode="auto", max_cost=5.0, api_key="fake-key", post=Stub([]))
        check(not svc2.pending(1, MD, queries) and svc2.results(1, MD, queries)[0].status == "located"
              and len(stub.bodies) == 1, "cache hit: no second call (same and new session)")
        svc2.close()
        off = GB.GeminiBoxService(ws, mode="offline", max_cost=5.0)
        check(off.results(1, MD, queries)[0].status == "located", "offline mode replays the cache")
        sm = off.page_map_summary(1)
        check(sm["source"] == "gemini" and sm["gemini_located"] == 2 and sm["gemini_total"] == 4
              and review.no_evidence_reason(sm) is None, "page summary: Gemini boxes 2 of 4, no-evidence banner off")
        other = [Query("متن", span_of("متن", 2))]
        r = off.results(1, MD, other)
        check(r[0].status == "unresolved" and r[0].reason == "gemini: offline, no cached boxes"
              and not off.pending(1, MD, other), "offline, uncached -> unresolved with reason, not pending")
        off.close()
        nokey = GB.GeminiBoxService(ws, mode="auto", max_cost=5.0, api_key="", post=_no_network)
        r = nokey.results(1, MD, other)
        check(r[0].reason == "no Gemini key" and not nokey.pending(1, MD, other), "no key -> unresolved, no Gemini key")
        nokey.schedule(1, MD, other, None)
        nokey.close()

        # -- budget exhaustion ----------------------------------------------------
        broke = GB.GeminiBoxService(ws, mode="auto", max_cost=0.0, api_key="fake-key", post=_no_network)
        broke.refine(1, MD, other)
        r = broke.results(1, MD, other)
        check(r[0].status == "unresolved" and r[0].reason == "budget exhausted", "budget exhausted -> no call, reason")
        broke.close()

        # -- HTTP errors: retry, no retry, access stop --------------------------------
        st = Stub([(503, {"error": {"message": "busy"}}), (503, {"error": {"message": "busy"}})])
        s = GB.GeminiBoxService(ws, mode="auto", api_key="fake-key", post=st)
        s.refine(1, MD, other)
        check(len(st.bodies) == 2 and s.results(1, MD, other)[0].reason == "gemini: http 503"
              and s.budget.snapshot()["uncertain"] > 0, "5xx retried once (2 attempts), uncertain cost kept")
        s.close()
        st = Stub([(400, {"error": {"message": "bad"}})])
        s = GB.GeminiBoxService(ws, mode="auto", api_key="fake-key", post=st)
        s.refine(1, MD, other)
        check(len(st.bodies) == 1 and s.results(1, MD, other)[0].reason == "gemini: http 400"
              and s.budget.snapshot()["spent"] == 0, "400 not retried, settled at $0")
        s.close()
        st = Stub([(403, {"error": {"message": "denied fake-key"}})])
        s = GB.GeminiBoxService(ws, mode="auto", api_key="fake-key", post=st)
        s.refine(1, MD, other)
        third = [Query("سطر", span_of("سطر"))]
        s.refine(1, MD, third)
        check(len(st.bodies) == 1 and s.results(1, MD, third)[0].reason.startswith("gemini: access refused"),
              "403 stops further calls for the session")
        s.close()
        st = Stub([answer([{"id": 1, "boxes": "not a list"}]), OSError("reset")])
        s = GB.GeminiBoxService(ws, mode="auto", api_key="fake-key", post=st)
        s.refine(1, MD, other)
        check(len(st.bodies) == 2 and s.results(1, MD, other)[0].reason == "gemini: transport OSError",
              "schema misfit retried, then transport failure reported")
        s.close()

        # -- review: /boxes payload, chip label, wait-for-boxes -------------------------
        hunk = {"id": 0, "issue_idx": None, "ctx_before": "", "old": "آزمایشی", "ctx_after": "", "new": "آزمونی",
                "unique": True, "occurrence": 1}
        stub = Stub([answer([{"id": 1, "boxes": [{"box_2d": [100, 300, 130, 500]}]}])])
        svc = GB.GeminiBoxService(ws, mode="auto", api_key="fake-key", post=stub)
        shutil.rmtree(tmp / "locate_evidence")
        with patch.object(review, "_REFINER", svc), \
             patch.object(review, "_page_box_inputs", return_value=({}, [], [hunk], MD, "hunks")), \
             patch.object(review, "_locate_queries_cached", return_value=(None,)):
            first = review._boxes_payload(ws, 1)
            check(first == {"pending": True}, "/boxes pending while the page is acquired in the background")
            done = threading.Event()
            t = threading.Thread(target=lambda: (review._wait_for_boxes(ws, [1], SimpleNamespace(is_alive=lambda: True)),
                                                 done.set()), daemon=True)
            t.start()
            done.wait(15)
            check(done.is_set() and len(stub.bodies) == 1, "--wait-for-boxes finishes after one background call")
            pl = review._boxes_payload(ws, 1)
            chip = pl["hunk_scores"]["0"]
            check(pl["pending"] is False and chip["label"] == "located (Gemini)" and chip["status"] == "located"
                  and chip["source"] == "gemini", "chip reads located (Gemini)")
            check(pl["boxes"][0]["source"] == "gemini" and pl["hunk_boxes"]["0"]["segments"][0]["x0"] == 0.3
                  and pl["evidence_banner"] is None and pl["page_map"]["gemini_located"] == 1,
                  "payload carries segments, no banner, Gemini summary")
        svc.close()

        # -- review factory and CLI -----------------------------------------------------
        with patch.object(review.llm, "load_env", lambda: None), \
             patch.object(hard_reader, "gemini_key", lambda: None):
            r = review._make_review_refiner(ws, "claude-sonnet-5", "auto", 5.0, "gemini")
            check(isinstance(r, GB.GeminiBoxService) and r.mode == "offline", "gemini engine without key -> offline")
            r.close()
        with patch.object(review.llm, "load_env", lambda: None), \
             patch.object(hard_reader, "gemini_key", lambda: "fake-key"):
            r = review._make_review_refiner(ws, "claude-sonnet-5", "auto", 5.0, "gemini")
            check(r.mode == "auto" and r.api_key == "fake-key" and r._post is _no_network,
                  "gemini engine with key -> auto (post function is the stubbed one)")
            r.close()
        runner = CliRunner()
        fake_ws = SimpleNamespace(slug="synthetic", meta={"page_count": 3}, pages_done=lambda: [1, 2, 3],
                                  review_dir=tmp / "review")
        with patch("farsi2epub.cli._load_workspace", return_value=fake_ws), \
             patch.object(review, "read_server_state", return_value=None), \
             patch.object(review, "run_review") as start:
            with patch.object(review, "default_bbox_engine", return_value="gemini"):
                res = runner.invoke(main, ["review", "synthetic"])
            check(res.exit_code == 0 and start.call_args.kwargs["bbox_engine"] == "gemini",
                  "default engine is gemini when a Gemini key is set")
            with patch.object(review, "default_bbox_engine", return_value="scan"):
                res = runner.invoke(main, ["review", "synthetic"])
            check(res.exit_code == 0 and start.call_args.kwargs["bbox_engine"] == "scan" and "No Gemini key" in res.output,
                  "default engine falls back to scan without a Gemini key, with a notice")
            res = runner.invoke(main, ["review", "synthetic", "--bbox-engine", "scan"])
            check(res.exit_code == 0 and start.call_args.kwargs["bbox_engine"] == "scan", "--bbox-engine scan still selectable")
            res = runner.invoke(main, ["review", "synthetic", "--bbox-engine", "gemini", "--bbox-model", "x"])
            check(res.exit_code == 0 and start.call_args.kwargs["bbox_engine"] == "gemini"
                  and "ignored" in res.output, "--bbox-engine gemini passed through; --bbox-model ignored notice")
        popen = {}
        states = iter([None, {"url": "http://127.0.0.1:1/"}])
        with patch.object(review, "read_server_state", side_effect=lambda w: next(states)), \
             patch.object(review.subprocess, "Popen", side_effect=lambda args, **k: popen.setdefault("args", args)):
            review.launch_review_background(fake_ws, bbox_engine="gemini")
        a = popen["args"]
        check(a[a.index("--bbox-engine") + 1] == "gemini" and "--bbox-model" not in a,
              "background re-exec passes --bbox-engine gemini, not --bbox-model")
        popen.clear()
        states = iter([None, {"url": "http://127.0.0.1:1/"}])
        with patch.object(review, "read_server_state", side_effect=lambda w: next(states)), \
             patch.object(review.subprocess, "Popen", side_effect=lambda args, **k: popen.setdefault("args", args)):
            review.launch_review_background(fake_ws)
        a = popen["args"]
        check(a[a.index("--bbox-engine") + 1] == "scan" and "--bbox-model" in a, "background default stays scan")
        print("GEMINI BOX ENGINE REGRESSIONS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
