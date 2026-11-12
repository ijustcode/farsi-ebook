"""Regression test for tests/transcription_experiments.py scoring, stitching,
budget guard and cache. Synthetic strings only; no API key, no books/.

    ./venv/bin/python tests/transcription_experiments_regression.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import transcription_experiments as te  # noqa: E402

ZW = "‌"
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL", msg)


def w(truth_lines, hyp_lines, folding):
    return te.wer(te.tokenize_lines(truth_lines), te.tokenize_lines(hyp_lines), folding)


def test_wer_foldings():
    t = ["این کتاب خوب است"]
    r = w(t, t, "strict")
    check(r["wer"] == 0 and r["n_truth"] == 4, f"identity {r}")
    # Arabic yeh/kaf and tatweel are folded even in strict
    r = w(["کتاب یک"], ["كتـــاب يك"], "strict")
    check(r["errors"] == 0, f"strict unicode fold {r}")
    # Typewriter join: truth has two words typed as one; hyp splits them.
    t = ["او وکتاب نامه"]
    h = ["او و کتاب نامه"]
    s, o, l = (w(t, h, f) for f in te.FOLDINGS)
    check(s["errors"] == 2 and o["errors"] == 2, f"join strict/ortho {s} {o}")
    check(l["errors"] == 0, f"join lenient {l}")
    # A joined form that also changes letters is NOT a join: still an error.
    check(w(["بنام"], ["به نام"], "lenient")["errors"] >= 1, "join with letter change")
    # Reverse: hyp fuses what truth splits (3-way join)
    check(w(["از آن جا رفت"], ["ازآنجا رفت"], "lenient")["errors"] == 0, "3-join lenient")
    # ZWNJ difference: strict counts, ortho/lenient do not
    t, h = ["میرود"], ["می" + ZW + "رود"]
    check(w(t, h, "strict")["errors"] == 1, "zwnj strict")
    check(w(t, h, "ortho")["errors"] == 0 and w(t, h, "lenient")["errors"] == 0, "zwnj ortho/lenient")
    # Punctuation spacing: strict counts the detached period, ortho drops it
    t, h = ["پایان ."], ["پایان."]
    check(w(t, h, "strict")["errors"] == 2, f"punct strict {w(t, h, 'strict')}")
    check(w(t, h, "ortho")["errors"] == 0, "punct ortho")
    # Inserted invented word
    t = ["مدیر شرکت باید امضا کند"]
    h = ["مدیر شرکت باید حتما امضا کند"]
    r = w(t, h, "lenient")
    check(r["errors"] == 1 and abs(r["wer"] - 0.2) < 1e-9, f"insert {r}")
    # Dropped line: 2 lines of 3 words, second missing -> 3 deletions of 6
    r = w(["الف ب ج", "د ه و"], ["الف ب ج"], "lenient")
    check(r["errors"] == 3 and r["n_truth"] == 6, f"dropped line {r}")
    # Page-number line ignored; owner's unreadable marker is a free wildcard
    r = w(["متن اول", "-۲-"], ["متن اول"], "strict")
    check(r["errors"] == 0, "page number dropped")
    r = w(["متن ⟦؟⟧ دوم"], ["متن هرچه دوم"], "strict")
    check(r["errors"] == 0 and r["n_truth"] == 2, f"truth wildcard {r}")
    # Markdown syntax stripped
    toks = te.tokenize_md("# عنوان\n\n**متن** اصلی --- دوم")
    check([t.text for t in toks] == ["عنوان", "متن", "اصلی", "دوم"], f"md strip {[t.text for t in toks]}")


def test_invented():
    truth = te.tokenize_lines(["مدیر شرکت باید امضا کند"])
    # one inserted unrelated word, one near-miss substitution (not invented)
    hyp = te.tokenize_lines(["مدیر شرکت بابد لازم امضا کند"])
    inv = te.invented_words(truth, hyp)
    check(len(inv) == 1, f"invented count {inv}")
    # unrelated substitution counts too; markers never do
    hyp = te.tokenize_lines(["مدیر شرکت معتبر امضا کند ⟦؟⟧"])
    check(len(te.invented_words(truth, hyp)) == 1, "invented sub + unk marker")


def test_markers():
    truth = te.tokenize_lines(["الف بب جج دد هه"])
    hyp = te.tokenize_lines(["الف ⟦بب؟⟧ ⟦ججج؟⟧ ⟦؟⟧ ههه"])
    m = te.marker_stats(truth, hyp)
    check(m["markers_unk"] == 1 and m["markers_guess"] == 2, f"marker counts {m}")
    check(m["guess_correct"] == 1 and m["guess_wrong"] == 1, f"guess accounting {m}")
    check(m["errors"] == 3 and m["errors_in_markers"] == 2 and m["silent_errors"] == 1, f"errors in markers {m}")
    # multi-word guess marker
    toks = te.tokenize_lines(["⟦دو کلمه؟⟧ سه"])
    check([t.marker for t in toks] == ["guess", "guess", None] and toks[0].marker_id == toks[1].marker_id,
          "multiword marker")


def test_t2():
    truth = te.tokenize_lines(["الف بب جج دد هه وو زز حح طط یی"])
    primary = te.tokenize_lines(["الف بب XX دد هه وو YY حح طط یی"])   # 2 errors
    secondary = te.tokenize_lines(["الف بب جج دد QQ وو YY حح طط یی"])  # disagrees at 3 and 5
    s = te.t2_stats(truth, primary, secondary)
    # flagged: XX (true error) and هه (false flag); YY error is shared -> missed
    check(s["errors"] == 2 and s["caught"] == 1 and abs(s["recall"] - 0.5) < 1e-9, f"t2 recall {s}")
    check(s["flagged_words"] == 2 and abs(s["flag_rate"] - 0.2) < 1e-9, f"t2 flag rate {s}")
    # deletion in primary caught by an insertion flag in secondary at the same gap
    primary = te.tokenize_lines(["الف بب دد"])
    secondary = te.tokenize_lines(["الف بب جج دد"])
    s = te.t2_stats(te.tokenize_lines(["الف بب جج دد"]), primary, secondary)
    check(s["errors"] == 1 and s["caught"] == 1 and s["flagged_gaps"] == 1, f"t2 gap {s}")


def test_vote3():
    P = te.tokenize_lines(["الف XX جج دد هه"])
    S1 = te.tokenize_lines(["الف بب جج YY هه"])
    S2 = te.tokenize_lines(["الف بب جج دد"])
    voted, unsure = te.vote3(P, S1, S2)
    # XX replaced by agreeing بب; دد kept (S1 differs, S2 agrees) but unsure; هه kept, unsure
    check([t.text for t in voted] == ["الف", "بب", "جج", "دد", "هه"], f"vote tokens {[t.text for t in voted]}")
    check(unsure == {3, 4}, f"vote unsure {unsure}")
    voted, _ = te.vote3(te.tokenize_lines(["الف ZZ بب"]), te.tokenize_lines(["الف بب"]), te.tokenize_lines(["الف بب"]))
    check([t.text for t in voted] == ["الف", "بب"], "vote drops doubly-deleted word")


def test_line_crop():
    import math
    import numpy as np
    # frame_point: zero angle is a translation; 90 degrees maps u to +y
    check(te.frame_point({"cx": 10, "cy": 20, "angle_deg": 0}, 5, -2) == (15, 18), "frame_point 0deg")
    x, y = te.frame_point({"cx": 0, "cy": 0, "angle_deg": 90}, 1, 0)
    check(abs(x) < 1e-9 and abs(y - 1) < 1e-9, "frame_point 90deg")
    # identity resample reproduces the source
    rng = np.random.default_rng(0)
    src = rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)
    out = te.sample_affine(src, 0, 0, (1, 0), (0, 1), 30, 20)
    check(np.array_equal(out, src), "identity resample")
    # a dark bar drawn along a 5-degree line comes out horizontal and centred
    H, W = 300, 600
    img = np.full((H, W, 3), 255, np.uint8)
    frame = {"cx": 300.0, "cy": 150.0, "angle_deg": 5.0}
    a = math.radians(5)
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx - 300) * math.cos(a) + (yy - 150) * math.sin(a)
    v = -(xx - 300) * math.sin(a) + (yy - 150) * math.cos(a)
    img[(np.abs(u) <= 200) & (np.abs(v) <= 10)] = 0
    line = {"frame": frame, "box": [-200, -10, 200, 10]}
    crop, meta = te.crop_line(img, 1.0, 0.0, 0.0, line)
    check(abs(meta["scale"] - te.LINE_SCALE_CAP) < 1e-9 and abs(meta["box_h_px"] - 20 * te.LINE_SCALE_CAP) < 1e-6,
          f"crop meta {meta}")
    rows = crop[..., 0].mean(axis=1)
    dark = np.where(rows < 80)[0]
    check(len(dark) > 0 and abs((dark[0] + dark[-1]) / 2 - crop.shape[0] / 2) <= 2, f"bar centred {dark[:3]}")
    check(rows[0] > 250 and rows[-1] > 250, "margins white")
    cols = crop[crop.shape[0] // 2, :, 0]
    check((cols[20:-20] < 80).mean() > 0.95, "bar straight along the whole strip")
    # context crop: taller, with red marks on the target rows only
    nb = [{"frame": frame, "box": [-200, -40, 200, -20]}, {"frame": frame, "box": [-200, 20, 200, 40]}]
    ctx, cm = te.crop_line(img, 1.0, 0.0, 0.0, line, tuple(nb), pitch=30.0)
    check(ctx.shape[0] > crop.shape[0], "context crop is taller")
    red = (ctx[..., 0] > 200) & (ctx[..., 1] < 40)
    rr = np.where(red.any(axis=1))[0]
    check(len(rr) > 0 and abs((rr[0] + rr[-1]) / 2 - ctx.shape[0] / 2) <= 3, "target marked")
    # PNG encode round-trip size
    import fitz
    pix = fitz.Pixmap(te.png_from_array(crop))
    check((pix.width, pix.height) == (crop.shape[1], crop.shape[0]), "png size")


def test_per_line_errors():
    truth = te.tokenize_lines(["الف بب", "جج دد", "هه وو"])
    hyp = te.tokenize_lines(["الف بب", "جج XX", "هه وو ZZ"])
    pl = te.per_line_errors(truth, hyp)
    check(pl["lines"] == 3 and pl["perfect_lines"] == 1 and pl["errors"] == 2, f"per line {pl}")
    check(pl["per_line"] == {0: 0, 1: 1, 2: 1}, f"per line map {pl['per_line']}")


def test_truth_wild():
    orig = te.truth_lines
    te.truth_lines = lambda pid: ["عنوان", "", "الف بب جج"]
    try:
        t = te.truth_tokens_wild("x", frozenset({0}))
    finally:
        te.truth_lines = orig
    check([x.text for x in t] == [te.MARK_UNK, "الف", "بب", "جج"], f"wild tokens {[x.text for x in t]}")
    # excluded line: an arm that skips it and one that reads it both score 0
    check(te.wer(t, te.tokenize_lines(["الف بب جج"]), "lenient")["errors"] == 0, "skip excluded")
    check(te.wer(t, te.tokenize_lines(["عنوان", "الف بب جج"]), "lenient")["errors"] == 0, "read excluded")


FAKE_KEY = "sk-test-SECRETSECRETSECRET123456"
FAKE_GKEY = "AIzaFAKEFAKEFAKEFAKEFAKE12345"


def _png(w, h):
    import numpy as np
    return te.Image(te.png_from_array(np.full((h, w, 1), 255, np.uint8)), "image/png", w, h, "t")


def _vreq(vendor, model, extra=None, rep=0):
    return te.Request("arm", "p", model, "SYS", ["Band 1:", _png(64, 32), "go"], te.BandsReading,
                      max_tokens=1000, rep=rep, extra=extra, vendor=vendor)


def test_vendor_request_building():
    url, body = te.OpenAIAdapter.build(_vreq("openai", "gpt-6-astra", {"effort": "low"}))
    check(url == "https://api.openai.com/v1/responses", "openai url")
    parts = body["input"][0]["content"]
    check([p["type"] for p in parts] == ["input_text", "input_image", "input_text"], "openai parts order")
    check(parts[1]["detail"] == "original" and parts[1]["image_url"].startswith("data:image/png;base64,"),
          "openai image detail original")
    check(body["instructions"] == "SYS" and body["reasoning"] == {"effort": "low"} and body["max_output_tokens"] == 1000,
          "openai fields")
    sch = body["text"]["format"]["schema"]
    check(body["text"]["format"]["strict"] is True and sch["additionalProperties"] is False
          and sch["required"] == ["bands"], "openai strict schema")
    check(te.OpenAIAdapter.headers(FAKE_KEY) == {"Authorization": "Bearer " + FAKE_KEY}, "openai header")
    check(FAKE_KEY not in json.dumps(body) and "key" not in url, "no key in body or url")
    url, body = te.GoogleAdapter.build(_vreq("google", "gemini-4-carbon", {"thinking_level": "low"}))
    check(url.endswith("/v1beta/interactions") and "key=" not in url, "google url has no key")
    check(body["input"][0] == {"type": "text", "text": "SYS"} and body["input"][2]["resolution"] == "ultra_high",
          "google items")
    check(body["generation_config"] == {"max_output_tokens": 1000, "thinking_level": "low"}, "google config")
    check("$ref" not in json.dumps(body["response_format"]["schema"]), "google schema inlined")
    check(te.GoogleAdapter.headers(FAKE_GKEY) == {"x-goog-api-key": FAKE_GKEY}, "google header")


def test_vendor_parsing():
    ok = {"status": "completed", "usage": {"input_tokens": 1200, "output_tokens": 300,
                                           "output_tokens_details": {"reasoning_tokens": 120}},
          "output": [{"type": "reasoning"}, {"type": "message", "content": [
              {"type": "output_text", "text": json.dumps({"bands": [{"band_index": 1, "lines": ["الف"]}]})}]}]}
    parsed, usage, prob = te.OpenAIAdapter.parse(ok)
    check(parsed["bands"][0]["lines"] == ["الف"] and prob is None, "openai parse")
    check(usage == {"input_tokens": 1200, "output_tokens": 300, "reasoning_tokens": 120}, f"openai usage {usage}")
    check(abs(te.vendor_cost("openai", "gpt-6-astra", usage) - (1200 * 10 + 300 * 50) / 1e6) < 1e-12, "openai cost")
    inc = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
           "usage": {"input_tokens": 10, "output_tokens": 1000}, "output": []}
    parsed, usage, prob = te.OpenAIAdapter.parse(inc)
    check(parsed is None and prob == "incomplete:max_output_tokens" and usage["output_tokens"] == 1000, "openai incomplete")
    ref = {"status": "completed", "usage": {}, "output": [{"type": "message", "content": [{"type": "refusal"}]}]}
    check(te.OpenAIAdapter.parse(ref)[2] == "refusal", "openai refusal")
    g = {"outputs": [{"type": "thought", "summary": "x"}, {"type": "text", "text": json.dumps({"ok": True})}],
         "usage": {"total_input_tokens": 500, "total_output_tokens": 40, "total_thought_tokens": 60}}
    parsed, usage, prob = te.GoogleAdapter.parse(g)
    check(parsed == {"ok": True} and usage == {"input_tokens": 500, "output_tokens": 100, "reasoning_tokens": 60},
          f"google parse {usage}")
    g2 = {"candidates": [{"content": {"parts": [{"text": "hm", "thought": True}, {"text": "{\"ok\": true}"}]}}],
          "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 2}}
    parsed, usage, _ = te.GoogleAdapter.parse(g2)
    check(parsed == {"ok": True} and usage["output_tokens"] == 5, "google generateContent-shape fallback")


def test_vendor_sizing_rules():
    # Anthropic high tier: 2576 edge and 4784 patches of 28 px
    check(te.vendor_fits("anthropic", "claude-sonnet-5", 2276, 1267), "anthropic band fits")
    check(not te.vendor_fits("anthropic", "claude-sonnet-5", 1821, 2576), "anthropic whole page is shrunk")
    # OpenAI gpt-6-astra, detail=original: 32 px patches, x1.2, <=30000 patches, <=65535 px
    check(te.vendor_image_tokens("openai", "gpt-6-astra", 2276, 1267) == math.ceil(72 * 40 * 1.2), "openai tokens")
    check(te.vendor_fits("openai", "gpt-6-astra", 2276, 3200), "openai astra keeps a whole native page")
    check(not te.vendor_fits("openai", "gpt-6-astra", 6000, 6000), "openai astra over 30000 patches")
    check(not te.vendor_fits("openai", "gpt-6.1-sol", 4000, 3000), "sol conservative 10000-patch rule")
    check(te.vendor_fits("google", "gemini-4-carbon", 2276, 1267) and
          not te.vendor_fits("google", "gemini-4-carbon", 3000, 3000), "google assumed pixel limit")
    check(te.vendor_image_tokens("google", "gemini-4-carbon", 10, 10) == 2240, "google fixed budget")


def test_redaction():
    t = te.redact(f"bad key {FAKE_KEY} and Bearer abc.def and {FAKE_GKEY} x-goog-api-key: zzz", FAKE_KEY)
    check(FAKE_KEY not in t and FAKE_GKEY not in t and "abc.def" not in t and "zzz" not in t, f"redact {t}")
    check(FAKE_KEY[:8] not in te.redact("prefix " + FAKE_KEY[:8] + "...", FAKE_KEY), "redact key prefix")


class _FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, body))
        return self.responses.pop(0)


def test_vendor_runner_flow():
    import tempfile
    good = (200, {"status": "completed", "usage": {"input_tokens": 1000, "output_tokens": 200},
                  "output": [{"type": "message", "content": [{"type": "output_text",
                              "text": json.dumps({"bands": [{"band_index": 1, "lines": ["x"]}]})}]}]})
    with tempfile.TemporaryDirectory() as d:
        http = _FakeHTTP([good])
        r = te.Runner(Path(d), cap=3.0, ledger_name="l2b.jsonl", http_post=http,
                      key_lookup=lambda v: FAKE_KEY if v == "openai" else None)
        req = _vreq("openai", "gpt-6-astra", {"effort": "low"})
        rec = r.call(req)
        check(len(http.calls) == 1 and rec["parsed"]["bands"][0]["lines"] == ["x"], "vendor call ok")
        rec2 = r.call(req)
        check(len(http.calls) == 1 and rec2["cache_hit"], "vendor cache hit is free")
        on_disk = "".join(p.read_text() for p in Path(d).rglob("*.json*"))
        check(FAKE_KEY not in on_disk and FAKE_KEY[:8] not in on_disk, "no key in ledger or cache")
        # no key: skipped, nothing called, nothing paid
        try:
            r.call(_vreq("google", "gemini-4-carbon"))
            check(False, "google without key should skip")
        except te.VendorSkipped:
            pass
        check(len(http.calls) == 1, "skip makes no call")
        # access error stops the vendor; the echoed key never reaches the ledger
        http2 = _FakeHTTP([(401, {"error": {"message": f"Incorrect API key provided: {FAKE_KEY}"}})])
        r2 = te.Runner(Path(d), cap=3.0, ledger_name="l2b_b.jsonl", http_post=http2, key_lookup=lambda v: FAKE_KEY)
        try:
            r2.call(_vreq("openai", "gpt-6-astra", rep=7))
            check(False, "401 should stop")
        except te.AccessError as exc:
            check(FAKE_KEY not in str(exc), "key not in exception text")
        try:
            r2.call(_vreq("openai", "gpt-6.1-sol", rep=7))
            check(False, "stopped vendor should refuse")
        except te.AccessError:
            pass
        check(len(http2.calls) == 1, "no call after vendor stopped")
        led = (Path(d) / "l2b_b.jsonl").read_text()
        check(FAKE_KEY not in led and "http:401" in led and r2.ledger.total() == 0.0, "401 ledger entry at $0")
        # budget guard applies to vendor calls too
        r3 = te.Runner(Path(d), cap=0.01, ledger_name="l2b_c.jsonl", http_post=_FakeHTTP([good]),
                       key_lookup=lambda v: FAKE_KEY)
        try:
            r3.call(_vreq("openai", "gpt-6-astra", rep=9))
            check(False, "guard should refuse")
        except te.BudgetExceeded:
            pass


def test_stitch():
    a = ["سطر یک اول", "سطر دو دوم", "سطر سه سوم", "سطر چهار چهارم"]
    b = ["سطر سه سوم", "سطر چهار چهارم", "سطر پنج پنجم", "سطر شش ششم"]
    out, warn = te.stitch_bands([a, b])
    check(out == a + b[2:] and not warn, f"duplicated overlap {out}")
    # a line missing from one band's copy of the overlap is kept from the other
    a2 = ["سطر یک اول", "سطر دو دوم", "سطر سه سوم"]          # clipped: no line 4
    b2 = ["سطر سه سوم", "سطر چهار چهارم", "سطر پنج پنجم"]
    out, _ = te.stitch_bands([a2, b2])
    check(out == ["سطر یک اول", "سطر دو دوم", "سطر سه سوم", "سطر چهار چهارم", "سطر پنج پنجم"], f"missing in A {out}")
    a3 = ["سطر یک اول", "سطر دو دوم", "سطر سه سوم", "سطر چهار چهارم"]
    b3 = ["سطر دو دوم", "سطر چهار چهارم", "سطر پنج پنجم"]     # B skipped line 3
    out, _ = te.stitch_bands([a3, b3])
    check(out == a3 + ["سطر پنج پنجم"], f"missing in B {out}")
    # slightly different readings of the same overlap line still match, kept once
    a4 = ["سطر یک اول", "سطر دو دوم"]
    b4 = ["سطر دو دوما", "سطر سه سوم"]
    out, _ = te.stitch_bands([a4, b4])
    check(len(out) == 3, f"fuzzy overlap {out}")
    # no overlap at all: concatenated with a warning
    out, warn = te.stitch_bands([["الف"], ["ب"]])
    check(out == ["الف", "ب"] and warn, "no overlap")


def test_linebreaks():
    t = ["الف بب جج", "دد هه وو", "زز حح"]
    s = te.line_stats(t, t)
    check(s["lines_break_correct"] == 3 and s["hyp_lines"] == 3, f"lines identity {s}")
    s = te.line_stats(t, ["الف بب", "جج دد هه وو", "زز حح"])
    check(s["lines_break_correct"] == 1, f"shifted break {s}")
    s = te.line_stats(t, ["الف بب جج دد هه وو", "زز حح"])
    check(s["lines_break_correct"] == 1 and s["hyp_lines"] == 2, f"merged {s}")


class _FakeUsage:
    input_tokens, output_tokens = 1000, 100


class _FakeClient:
    def __init__(self):
        self.calls = 0
        self.messages = self

    def parse(self, **kw):
        self.calls += 1
        return SimpleNamespace(usage=_FakeUsage(), parsed_output=kw["output_format"](lines=["x"]),
                               stop_reason="end_turn")


def test_budget_and_cache():
    with tempfile.TemporaryDirectory() as d:
        fake = _FakeClient()
        runner = te.Runner(Path(d), client_factory=lambda: fake, cap=0.05)
        req = te.Request("t", "p", te.config.MODEL_STRONG, "sys", ["hello"], te.LineReading,
                         max_tokens=1000)
        rec = runner.call(req)
        check(fake.calls == 1 and not rec["cache_hit"], "first call paid")
        spent = runner.ledger.total()
        check(abs(spent - (1000 * 2 + 100 * 10) / 1e6) < 1e-9, f"ledger cost {spent}")
        rec2 = runner.call(req)
        check(fake.calls == 1 and rec2["cache_hit"] and runner.ledger.total() == spent, "cache hit is free")
        # a call whose worst case would pass the cap is refused before calling
        big = te.Request("t", "p", te.config.MODEL_STRONG, "sys", ["other"], te.LineReading, max_tokens=8000)
        try:
            runner.call(big)
            check(False, "budget guard did not refuse")
        except te.BudgetExceeded:
            pass
        check(fake.calls == 1, "no call after refusal")
        # rep changes the key (a deliberate fresh sample)
        req_rep = te.Request("t", "p", te.config.MODEL_STRONG, "sys", ["hello"], te.LineReading,
                             max_tokens=1000, rep=1)
        check(te.request_key(req_rep.key_dict()) != te.request_key(req.key_dict()), "rep in key")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    if FAILS:
        print(f"{len(FAILS)} failure(s)")
        sys.exit(1)
    print("transcription_experiments_regression: all checks passed")
