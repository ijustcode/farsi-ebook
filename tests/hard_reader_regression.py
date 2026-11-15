"""Hard-page reader (farsi2epub/hard_reader.py) and its hook in transcribe.

Offline: no API, no key. Three parts:
  1. The production request is the measured configuration (arm ``t1r_opus`` of
     tests/transcription_experiments.py): instruction, schema, band rows,
     stitching, model, output limit and effort are compared with the
     experiment script. With the private books present, the full request for
     hadpi p1-p2 (image bytes included) is compared too; skipped otherwise.
  2. Printed lines -> Markdown: the page's words are the reader's words in
     order, breaks and headings come from the first pass, fallbacks.
  3. transcribe._transcribe_one: which pages are re-read, what is stored.
"""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from farsi2epub import config, hard_reader as HR, llm, transcribe  # noqa: E402
from farsi2epub.config import MODEL_HARD, MODEL_STRONG, RES_HI  # noqa: E402
import transcription_experiments as TE  # noqa: E402


def test_same_configuration():
    for n in (1, 2, 3, 5):
        assert HR.system_prompt(n) == TE.BANDS_READ_SYSTEM(n), n
    assert HR.BandsReading.model_json_schema() == TE.BandsReading.model_json_schema()
    assert (MODEL_HARD, HR.EFFORT, HR.BAND_OVERLAP_PX) == (TE.MODEL_OPUS, TE.OPUS_EFFORT, TE.BAND_OVERLAP_PX)
    assert config.PRICES[MODEL_HARD] == TE.PRICES_EXTRA[TE.MODEL_OPUS]
    for h, w in ((2576, 1821), (4032, 3024), (3000, 2250), (1568, 1100), (900, 2576), (5000, 1200)):
        assert HR.band_rows(h, w) == TE._band_rows(h, w), (h, w)
    bands = [["الف ب ج", "دال ذال", "سطر سوم است", "سطر چهارم"],
             ["سطر سوم است", "سطر چهارم", "سطر پنجم", "خط ششم"],
             [], ["خط ششم.", "هفتم"], ["بی ربط"]]
    assert HR.stitch_bands(bands) == TE.stitch_bands(bands)
    assert HR.stitch_bands(bands)[0] == ["الف ب ج", "دال ذال", "سطر سوم است", "سطر چهارم", "سطر پنجم",
                                         "خط ششم", "هفتم", "بی ربط"]
    for s in ("كتاب‌هاي ١٢", "«مؤسّسه»،", "بـــهتر"):
        assert HR.strict_fold(s) == TE.strict_fold(s) and HR.ortho_fold_token(s) == TE.ortho_fold_token(s)


def test_same_request_on_real_pages():
    slug = TE.HADPI
    if not (ROOT / "books" / slug / "source.pdf").is_file():
        return "skipped (no books/)"
    import yaml
    meta = yaml.safe_load((ROOT / "books" / slug / "book.yaml").read_text())
    for pid in ("h1", "h2"):            # development pages only
        page = TE.PAGES[pid][1]
        ref = TE.build_request("t1r_opus", pid)
        hires = (ROOT / "books" / slug / "pages" / "hires" / f"{page:04d}.png").read_bytes()
        req = HR.build_request(ROOT / "books" / slug / "source.pdf", page, meta["page_count"],
                               meta.get("title_fa"), hires)
        mine = {"model": req["model"], "max_tokens": req["max_tokens"], "system": req["system"],
                "content": [{"image": c.sha, "mt": c.media_type} if isinstance(c, HR.BandImage) else {"text": c}
                            for c in req["content"]],
                "schema": req["schema"].model_json_schema(), "thinking": req["kwargs"], "rep": 0,
                "extra": req["kwargs"]}
        assert mine == ref.key_dict(), pid
    return "identical to the cached experiment requests"


def test_markdown():
    lines, guessed, unread = HR.plain_lines(
        ["فصل اول", "ماده ۱) نام ⟦شرکت؟⟧ : شرکت ⟦؟⟧ است", "- ۲ -", "و ⟦دو کلمه؟⟧ دیگر ."])
    assert lines == ["فصل اول", "ماده ۱) نام شرکت : شرکت [؟] است", "و دو کلمه دیگر ."]
    assert guessed == ["شرکت", "دو کلمه"] and unread == 1

    printed = ["اساسنامه شرکت", "فصل اول", "ماده ۱) نام شرکت: شرکت نمونه است و مدت",
               "آن نامحدود میباشد .", "ماده ۲) موضوع شرکت عبارتست از تولید و فروش",
               "کالاهای صنعتی در داخل کشور ."]
    # first pass: wrong words, modern spelling, a different heading split
    first = ("# اساسنامه شرکت\n\n## فصل اول\n\n### ماده ۱) نام شرکت\n\nشرکت نمونه است و مدت آن نامحدود می‌باشد.\n\n"
             "ماده ۲) موضوع شرکت عبارت است از تولید و فروش کالاهای صنعتی در خارج کشور.\n")
    md, headings, rep = HR.lines_to_markdown(printed, first)
    assert md.replace("#", " ").split() == " ".join(printed).split(), md      # the reader's words, in order
    assert rep["structure"] == "first_pass" and rep["aligned"] > 0.8
    blocks = md.strip().split("\n\n")
    assert blocks[0] == "# اساسنامه شرکت" and blocks[1] == "## فصل اول", blocks
    assert blocks[2] == "### ماده ۱) نام شرکت:" and blocks[3].startswith("شرکت نمونه") and blocks[3].endswith("میباشد ."), blocks
    assert blocks[4].startswith("ماده ۲)") and blocks[4].endswith("داخل کشور ."), blocks
    assert headings == ["اساسنامه شرکت", "فصل اول", "ماده ۱) نام شرکت:"]
    assert "میباشد" in md and "می‌باشد" not in md and "داخل" in md and "خارج" not in md

    # a first pass that has nothing to do with the page: one paragraph per printed line
    md2, h2, rep2 = HR.lines_to_markdown(printed, "# عنوان دیگر\n\nمتنی که هیچ ربطی به این صفحه ندارد و کاملا ساختگی است.\n")
    assert rep2["structure"] == "printed_lines" and h2 == [] and md2.strip().split("\n\n") == printed
    # a heading may not swallow body text
    md3, h3, _ = HR.lines_to_markdown(["عنوان", "متن اول صفحه که طولانی است و ادامه دارد تا پایان سطر"],
                                      "# عنوان\n")
    assert h3 == [] and "#" not in md3
    assert HR.structure_blocks("```verse\nالف --- ب\n```") is None
    # footnotes: references leave the text, definitions are their own blocks
    refs = []
    assert HR.structure_blocks("متن اول[^1] دوم\n\n[^1]: پانوشت", refs) == [("p", "متن اول دوم"), ("fn1", "پانوشت")]
    assert refs == [(0, 1, 1)]
    fn_first = "# عنوان\n\nاین متن صفحه است که واژه‌ای[^3] دشوار دارد و ادامه پیدا می‌کند.\n\n[^3]: دشوار = سخت"
    # the reader prints the number after the word and "N ـ text" at the foot
    md, heads, rep = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای ۳ دشوار", "دارد و ادامه پیدا میکند .", "۳ ـ دشوار = سخت"], fn_first)
    assert md == "# عنوان\n\nاین متن صفحه است که واژهای[^3] دشوار دارد و ادامه پیدا میکند .\n\n[^3]: دشوار = سخت\n", md
    assert heads == ["عنوان"] and rep["footnotes"]["refs_without_digit"] == 0 and rep["footnotes"]["defs_from_first_pass"] == 0
    # digit printed against the word; and no digit at all (the first pass's reference is kept)
    md, _, _ = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای۳ دشوار", "دارد و ادامه پیدا میکند .", "۳ ـ دشوار = سخت"], fn_first)
    assert "واژهای[^3] دشوار" in md and "۳" not in md, md
    md, _, rep = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای دشوار", "دارد و ادامه پیدا میکند .", "۳ ـ دشوار = سخت"], fn_first)
    assert "واژهای[^3] دشوار" in md and rep["footnotes"]["refs_without_digit"] == 1, md
    # the reader lost the definition line: the first pass's definition stands, so no reference dangles
    md, _, rep = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای ۳ دشوار", "دارد و ادامه پیدا میکند ."], fn_first)
    assert md.endswith("[^3]: دشوار = سخت\n") and rep["footnotes"]["defs_from_first_pass"] == 1, md
    # a footnote only the reader saw (live sample, bachehaye p64)
    plain = "# عنوان\n\nاین متن صفحه است که واژه‌ای دشوار دارد و ادامه پیدا می‌کند."
    md, _, rep = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای ۱ دشوار", "دارد و ادامه پیدا میکند .", "۱ ـ دشوار = سخت"], plain)
    assert md == "# عنوان\n\nاین متن صفحه است که واژهای[^1] دشوار دارد و ادامه پیدا میکند .\n\n[^1]: دشوار = سخت\n", md
    assert rep["footnotes"]["from_reader"] == 1
    # a numbered bottom line with no such number in the body is ordinary text, as is a number that starts a line
    md, _, rep = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای دشوار", "دارد و ادامه پیدا میکند .", "۱ ـ دشوار = سخت"], plain)
    assert "[^" not in md and "۱ ـ دشوار = سخت" in md and "footnotes" not in rep, md
    md, _, _ = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای دشوار", "۱ دارد و ادامه پیدا میکند .", "۱ ـ دشوار = سخت"], plain)
    assert "[^" not in md, md
    # the reader's numbering wins: the first pass numbered the note 2 and gave no definition
    md, _, rep = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای ۱ دشوار", "دارد و ادامه پیدا میکند .", "۱ ـ دشوار = سخت", "ادامه پانوشت"],
                                      plain.replace("واژه‌ای", "واژه‌ای[^2]"))
    assert "واژهای[^1] دشوار" in md and "[^2]" not in md and md.endswith("[^1]: دشوار = سخت ادامه پانوشت\n"), md
    # real shapes (bachehaye p78, p25): digits printed against the word and misread (1, 1, 2 for 1, 2, 3);
    # a footnote referred to twice; a definition that runs on to a second line
    body = "# عنوان\n\nاین متن صفحه است که واژه‌ای دشوار دارد و ادامه پیدا می‌کند و باز هم ادامه دارد تا پایان."
    foot = ["۱ - متن = نوشته", "۲ - واژه = کلمه ای که", "معنی دارد", "۳ - ادامه = دنباله"]
    md, _, rep = HR.lines_to_markdown(["عنوان", "این متن۱ صفحه است که واژهای۱ دشوار", "دارد و ادامه۲ پیدا میکند و باز هم ادامه دارد تا پایان ."] + foot, body)
    assert "متن[^1] صفحه" in md and "واژهای[^2] دشوار" in md and "ادامه[^3] پیدا" in md, md
    assert md.endswith("[^1]: متن = نوشته\n[^2]: واژه = کلمه ای که معنی دارد\n[^3]: ادامه = دنباله\n"), md
    assert rep["footnotes"]["refs"] == 3 and rep["footnotes"]["defs"] == 3
    md, _, rep = HR.lines_to_markdown(["عنوان", "این متن ۱ صفحه است که واژهای ۲ دشوار", "دارد و ادامه ۳ پیدا میکند و باز هم ادامه ۳ دارد تا پایان ."] + foot, body)
    assert "ادامه[^3] پیدا" in md and "هم ادامه[^3] دارد" in md and rep["footnotes"]["refs"] == 4, md
    # a first-pass reference with no digit from the reader and no definition anywhere is not kept
    md, _, rep = HR.lines_to_markdown(["عنوان", "این متن صفحه است که واژهای دشوار", "دارد و ادامه پیدا میکند ."], plain.replace("واژه‌ای", "واژه‌ای[^1]"))
    assert "[^" not in md and rep["footnotes"]["refs_dropped"] == 1, md
    assert HR.structure_blocks("| a | b |") is None
    assert HR.structure_blocks("> نقل\n\n- مورد\n\n**پررنگ**") == [("p", "نقل"), ("p", "مورد"), ("p", "پررنگ")]
    assert HR.lines_to_markdown([], first)[0] == ""


FIRST = "# عنوان سند\n\nاین متن صفحه است که خوانده شد و چند غلط دارد.\n"
PRINTED = ["عنوان سند", "این متن صفحه است که خوانده شد", "و چند ⟦غلط؟⟧ ⟦؟⟧ دارد ."]


def _run(conf, flags, embedded=None, hard_pages=True, first=FIRST, fail=False):
    calls = []

    def fake_transcribe(client, png, model_, n, page_count, title_hint, extra_hint=None):
        calls.append(("first", model_))
        return (llm.PageTranscription(is_blank=False, text_md=first, headings=["عنوان سند"] if first is FIRST else [], flags=list(flags),
                                      starts_mid_paragraph=False, continues_next=True, confidence=conf),
                {"input_tokens": 10, "output_tokens": 5}, 0.03)

    def fake_read(client, pdf, n, page_count, title, png):
        calls.append(("hard", n))
        if fail:
            raise RuntimeError("credit balance is too low")
        return list(PRINTED), {"source": "hires_render", "bands": 2}, {"input_tokens": 100, "output_tokens": 50}, 0.09

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "0001.first.md").write_text("stale")
        ws = SimpleNamespace(slug="synthetic", meta={"title_fa": None, "source_type": "scanned"},
                             pdf_path=root / "source.pdf", text_dir=root,
                             page_md_path=lambda n: root / f"{n:04d}.md",
                             page_meta_path=lambda n: root / f"{n:04d}.json")
        state = transcribe.RunState()
        with patch.object(llm, "transcribe_page", side_effect=fake_transcribe), \
             patch.object(HR, "read_page", side_effect=fake_read), \
             patch.object(transcribe, "_page_image", side_effect=lambda ws_, n, res: res.encode()), \
             patch.object(transcribe, "extract_embedded_text", return_value=embedded), \
             patch.object(transcribe, "extract_heading_candidates", return_value=[]), \
             patch.object(transcribe.qc, "record_event", return_value=None):
            line = transcribe._transcribe_one(ws, None, 1, 5, MODEL_STRONG, RES_HI, state, hard_pages)
        side = json.loads((root / "0001.json").read_text())
        md = (root / "0001.md").read_text()
        firstmd = (root / "0001.first.md").read_text() if (root / "0001.first.md").is_file() else None
        return side, md, firstmd, calls, state, line


def test_transcribe_hook():
    # hard page, no oracle: re-read, words replaced, first pass kept, still for review
    side, md, firstmd, calls, state, line = _run(0.55, [])
    assert calls == [("first", MODEL_STRONG), ("hard", 1)]
    assert md == "# عنوان سند\n\nاین متن صفحه است که خوانده شد و چند غلط [؟] دارد .\n", md
    assert firstmd == FIRST
    assert side["model_used"] == MODEL_HARD and side["needs_review"] is True
    hr = side["hard_reader"]
    assert hr["adopted"] and hr["guessed_words"] == ["غلط"] and hr["unreadable_words"] == 1
    assert hr["first_pass"] == {"confidence": 0.55, "flags": [], "text": "0001.first.md"}
    assert "illegible" in side["flags"] and side["headings"] == ["عنوان سند"] and side["continues_next"] is True
    assert side["confidence"] == 0.55 and abs(side["cost_usd"] - 0.12) < 1e-9
    assert side["usage"] == {"input_tokens": 110, "output_tokens": 55}
    assert abs(state.total_cost - 0.12) < 1e-9 and state.hard_read == 1 and "hard-page reader" in line
    # the illegible flag alone triggers it
    assert _run(0.9, ["illegible"])[3][-1] == ("hard", 1)
    # confident page: untouched, and a stale first-pass copy is removed
    side, md, firstmd, calls, state, _ = _run(0.9, [])
    assert calls == [("first", MODEL_STRONG)] and side["hard_reader"] is None and md == FIRST and firstmd is None
    assert side["model_used"] == MODEL_STRONG and side["needs_review"] is False
    # "all": a confident page without an oracle is re-read too, and stays for review
    side, md, firstmd, calls, _, _ = _run(0.9, [], hard_pages="all")
    assert calls[-1] == ("hard", 1) and side["hard_reader"]["adopted"] and side["needs_review"] is True
    assert ("hard", 1) not in _run(0.9, [], embedded=" ".join(["واژه"] * 40), hard_pages="all")[3]
    # a page with an embedded-text oracle is never re-read, even when flagged
    emb = " ".join(["واژه"] * 40)
    side, md, _, calls, _, _ = _run(0.55, [], embedded=emb)
    assert ("hard", 1) not in calls and side["hard_reader"] is None and side["needs_review"] is True
    # switched off
    side, md, _, calls, _, _ = _run(0.55, [], hard_pages=False)
    assert calls == [("first", MODEL_STRONG)] and side["hard_reader"] is None and md == FIRST
    # verse in the first pass: not re-read, no call paid
    verse = "```verse\nمصراع اول --- مصراع دوم\n```\n"
    side, md, _, calls, _, _ = _run(0.55, [], first=verse)
    assert calls == [("first", MODEL_STRONG)] and side["hard_reader"]["adopted"] is False and md == verse
    # the reader fails: the first pass stands, the reason is stored
    side, md, firstmd, calls, state, _ = _run(0.55, [], fail=True)
    assert md == FIRST and firstmd is None and side["model_used"] == MODEL_STRONG and side["needs_review"] is True
    assert side["hard_reader"]["adopted"] is False and "credit" in side["hard_reader"]["reason"]
    assert abs(state.total_cost - 0.03) < 1e-9


def test_book_rule():
    def side(conf, hard=None, embedded=None, blank=False, reviewed=None, **kw):
        d = {"confidence": conf, "flags": [], "is_blank": blank, "hard_reader": hard, "needs_review": conf < 0.70,
             "headings": [], "usage": {"input_tokens": 10, "output_tokens": 5}, "cost_usd": 0.03,
             "validators": {"embedded_similarity": embedded, "issues": []}, **kw}
        if reviewed:
            d["reviewed"] = reviewed
        return d
    D = transcribe.hard_book_decision
    took = {"adopted": True}
    # a fifth or more of the no-oracle pages fired: the rest become candidates
    d = D({1: side(0.5, took), 2: side(0.6, took), 3: side(0.75), 4: side(0.9)})
    assert d["hard"] and d["fired"] == 2 and d["counted"] == 4 and d["candidates"] == [3, 4], d
    assert D({1: side(0.5), 2: side(0.75), 3: side(0.75), 4: side(0.9), 5: side(0.9)})["hard"]
    # under a fifth: nothing; pages with an oracle and blank pages are not counted
    assert not D({n: side(0.5 if n == 1 else 0.9) for n in range(1, 7)})["hard"]
    d = D({1: side(0.5), 2: side(0.6), 3: side(0.75), 4: side(0.5, embedded=0.99), 5: side(0.1, blank=True)})
    assert d["hard"] and d["counted"] == 3 and d["candidates"] == [1, 2, 3], d
    # too few pages to call a book hard
    assert not D({1: side(0.5), 2: side(0.5)})["hard"]
    # the stored first-pass verdict wins over a recomputation; a re-read page's added flag does not count
    assert not D({1: side(0.5, first_pass_review=False), 2: side(0.9), 3: side(0.9)})["hard"]
    assert D({1: side(0.9, first_pass_review=True), 2: side(0.9), 3: side(0.9)})["hard"]
    reread = dict(side(0.9, {"adopted": True, "first_pass": {"flags": []}}), flags=["illegible"])
    assert D({1: reread, 2: side(0.9), 3: side(0.9)})["fired"] == 0

    # end to end on stored first passes: page 3 (0.75) is re-read, a reviewed page and a page outside the run are not
    calls = []

    def fake_read(client, pdf, n, page_count, title, png):
        calls.append(n)
        return list(PRINTED), {"source": "hires_render", "bands": 2}, {"input_tokens": 100, "output_tokens": 50}, 0.09

    with tempfile.TemporaryDirectory() as d_:
        root = Path(d_)
        book = {1: side(0.5, took), 2: side(0.6, took), 3: side(0.75), 4: side(0.8, reviewed="accepted"), 5: side(0.65)}
        for n, sc in book.items():
            (root / f"{n:04d}.json").write_text(json.dumps(sc))
            (root / f"{n:04d}.md").write_text(FIRST)
        ws = SimpleNamespace(slug="synthetic", meta={"title_fa": None}, pdf_path=root / "source.pdf",
                             page_md_path=lambda n: root / f"{n:04d}.md", page_meta_path=lambda n: root / f"{n:04d}.json",
                             pages_done=lambda: sorted(book))
        state = transcribe.RunState()
        with patch.object(HR, "read_page", side_effect=fake_read), \
             patch.object(transcribe, "_page_image", side_effect=lambda ws_, n, res: res.encode()):
            transcribe._book_rule(ws, None, [1, 2, 3, 4], 5, state, None, 2)
            assert calls == [3], calls
            s3 = json.loads((root / "0003.json").read_text())
            hr = s3["hard_reader"]
            assert hr["adopted"] and hr["trigger"] == "book" and s3["model_used"] == MODEL_HARD
            assert s3["needs_review"] is True and s3["confidence"] == 0.75 and s3["first_pass_review"] is False
            assert abs(s3["cost_usd"] - 0.12) < 1e-9 and s3["usage"]["input_tokens"] == 110
            assert (root / "0003.first.md").read_text() == FIRST and (root / "0003.md").read_text() != FIRST
            assert (root / "0003.reader.txt").read_text().split("\n")[:2] == PRINTED[:2]
            assert json.loads((root / "0004.json").read_text())["hard_reader"] is None
            assert (root / "0005.md").read_text() == FIRST
            assert state.hard_read == 1 and abs(state.total_cost - 0.09) < 1e-9
            # a second run changes nothing; a spent budget stops the rule before any call
            transcribe._book_rule(ws, None, [1, 2, 3, 4], 5, state, None, 2)
            assert calls == [3]
            transcribe._book_rule(ws, None, [5], 5, transcribe.RunState(total_cost=1.0), 1.0, 2)
            assert calls == [3]


def run():
    failures = 0
    for fn in (test_same_configuration, test_same_request_on_real_pages, test_markdown, test_transcribe_hook, test_book_rule):
        try:
            note = fn()
            print(f"ok   {fn.__name__}" + (f": {note}" if note else ""))
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"{'PASS' if not failures else 'FAIL'}: {failures} failure(s)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
