#!/usr/bin/env python3
"""Regression test for the synthetic page generator (tests/synth_pages.py).

Checks:
  1. determinism: same spec + seed => identical image hash and truth hash;
  2. transform algebra: inverse(forward(p)) == p for the full photo transform,
     and a pure rotation maps a known rectangle to the analytically expected
     corners;
  3. image/truth agreement under a pure rotation: a black rectangle drawn in
     the clean image lands, after warp_image, exactly where the forward-mapped
     truth quad says (centroid and extent within 1 px), and on a generated
     rotated page >= 99% of words' warped ink lies inside their warped polygon;
  4. text-error injection: deterministic, exact op count, consistent mapping;
  5. tracked manifest (tests/data/synth_manifest.json): two v1 entries and the
     two v2 preset entries regenerate to the recorded hashes, when the font
     files and PyMuPDF version match (otherwise reported as SKIP, not a pass);
  6. pipeline v2 (generator degradations for the hard typewritten-photo
     profile): off by default and for every v1 preset; deterministic by seed;
     each degradation alone keeps >= 99% of words' ink inside their truth
     box/polygon; specks and typed underlines are recorded as non-text and
     change no word or line; word-gap records agree with the id map and with
     the rendered text layer; overlay coverage agrees with a recomputation;
     the per-line baseline warp inverts exactly and the hard preset reaches
     and exceeds 3.3 deg; the invented text pool does not repeat phrases.

Run: ANTHROPIC_BASE_URL=http://127.0.0.1:9 ./venv/bin/python tests/synth_pages_regression.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import fitz
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_pages as S  # noqa: E402
import synth_smoke_1a as T  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def t_determinism() -> None:
    print("1. determinism")
    spec = S.make_spec("modern_naskh", preset="photo", seed=7)
    a, b = S.generate(spec), S.generate(spec)
    check(a["truth"]["image"]["sha256"] == b["truth"]["image"]["sha256"], "same image hash on regeneration")
    check(a["truth"]["truth_sha256"] == b["truth"]["truth_sha256"], "same truth hash on regeneration")
    check(np.array_equal(a["image"], b["image"]), "pixel-identical images")
    c = S.generate(S.make_spec("modern_naskh", preset="photo", seed=8))
    check(c["truth"]["image"]["sha256"] != a["truth"]["image"]["sha256"], "different seed gives a different page")


def t_transform() -> None:
    print("2. transform algebra")
    W, H = 1821, 2576
    tf = S.PageTransform(W, H, dict(S.DEGRADE_PRESETS["photo"]), seed=3)
    xs = np.random.default_rng(0).uniform(0, W, 2000)
    ys = np.random.default_rng(1).uniform(0, H, 2000)
    fx, fy = tf.forward(xs, ys)
    bx, by = tf.inverse(fx, fy)
    err = float(np.max(np.hypot(bx - xs, by - ys)))
    check(err < 1e-3, f"inverse(forward(p)) == p for photo transform (max err {err:.2e} px)")
    a = 2.0
    rot = S.PageTransform(W, H, {"rotate_deg": a}, seed=0)
    rect = np.array([[400, 600], [900, 600], [900, 700], [400, 700]], float)
    got = np.stack(rot.forward(rect[:, 0], rect[:, 1]), axis=1)
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    cx, cy = W / 2, H / 2
    exp = np.array([[cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)] for x, y in rect])
    check(float(np.max(np.abs(got - exp))) < 1e-9, "pure rotation maps a known rectangle to the expected corners")


def t_rotation_image() -> None:
    print("3. image and truth move together under a pure rotation")
    W, H = 1821, 2576
    img = np.ones((H, W), np.float32)
    x0, y0, x1, y1 = 400, 600, 900, 700
    img[y0:y1, x0:x1] = 0.0
    rot = S.PageTransform(W, H, {"rotate_deg": 2.0}, seed=0)
    out = S.warp_image(img, rot, fill=1.0)
    ys, xs = np.nonzero(out < 0.5)
    cen = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
    ex = np.array(rot.forward(np.array([(x0 + x1) / 2]), np.array([(y0 + y1) / 2]))).ravel()
    check(float(np.hypot(*(cen - ex))) < 1.0, f"dark-rectangle centroid at forward-mapped centre (off by {np.hypot(*(cen - ex)):.2f} px)")
    q = np.stack(rot.forward(np.array([x0, x1, x1, x0], float), np.array([y0, y0, y1, y1], float)), axis=1)
    ok = (abs(xs.min() - q[:, 0].min()) <= 1.0 and abs(xs.max() + 1 - q[:, 0].max()) <= 1.0
          and abs(ys.min() - q[:, 1].min()) <= 1.0 and abs(ys.max() + 1 - q[:, 1].max()) <= 1.0)
    check(ok, "dark-rectangle extent matches forward-mapped quad within 1 px")
    res = S.generate(S.make_spec("modern_naskh", preset="clean", seed=4, degrade={"rotate_deg": 2.0}), keep_arrays=True)
    d = T.check_degraded(res)
    check(d["frac"] >= 0.99, f"rotated page: {100 * d['frac']:.2f}% of words' warped ink inside warped polygon")


def t_injection() -> None:
    print("4. text-error injection")
    words = " ".join(S.PROSE).split()
    for rate in (0.05, 0.15):
        a = S.inject_text_errors(words, rate, seed=9)
        b = S.inject_text_errors(words, rate, seed=9)
        check(a == b, f"rate {rate}: deterministic by seed")
        check(len(a["ops"]) == round(rate * len(words)), f"rate {rate}: {len(a['ops'])} ops for {len(words)} words")
        kept = [i for i, st in enumerate(a["status"]) if st == "kept"]
        mapped = [i for i in a["src_index"] if i is not None]
        consistent = (len(a["status"]) == len(words)
                      and all(a["words"][k] == words[i] for k, i in enumerate(a["src_index"])
                              if i is not None and a["status"][i] == "kept")
                      and set(kept) <= set(mapped))
        check(consistent, f"rate {rate}: mapping consistent (kept words unchanged and mapped)")
    c = S.inject_text_errors(words, 0.15, seed=10)
    check(c["words"] != S.inject_text_errors(words, 0.15, seed=9)["words"], "different seed, different corruption")


def t_manifest() -> None:
    print("5. tracked manifest")
    if not S.MANIFEST_PATH.exists():
        print("  SKIP no manifest"); return
    man = json.loads(S.MANIFEST_PATH.read_text(encoding="utf-8"))
    if man.get("pymupdf") != fitz.VersionBind:
        print(f"  SKIP PyMuPDF {fitz.VersionBind} != manifest {man.get('pymupdf')}"); return
    picks = [e for e in man["entries"] if e["name"].startswith(("modern_naskh", "typewriter")) and "photo_s" in e["name"]][:2]
    picks += [e for e in man["entries"] if e["spec"].get("preset") in ("tw_photo_medium", "tw_photo_hard")]
    if len(picks) < 4:
        check(False, f"manifest has the two v1 and two v2 preset entries ({len(picks)} found)")
    for e in picks:
        drift = [f["path"] for f in e["fonts"] if not Path(S._abs(f["path"])).exists()
                 or S.sha256_file(S._abs(f["path"])) != f["sha256"]]
        if drift:
            print(f"  SKIP {e['name']}: font file changed or missing: {drift}"); continue
        res = S.generate(e["spec"])
        check(res["truth"]["image"]["sha256"] == e["image_sha256"], f"{e['name']}: image hash matches manifest")
        check(res["truth"]["truth_sha256"] == e["truth_sha256"], f"{e['name']}: truth hash matches manifest")


def _memo(name, degrade, preset="clean", seed=5):
    return S.make_spec("typewriter", preset=preset, seed=seed, layout="memo", text_source="pool",
                       leading=1.75, degrade=degrade, name=name)


def t_v2_off_by_default() -> None:
    print("6a. v2 off by default")
    check(all(not S.DEGRADE_KEYS[k] for k in S._V2_SWITCHES), "every v2 switch defaults to off")
    v1 = [p for p in S.DEGRADE_PRESETS if p not in ("tw_photo_hard", "tw_photo_medium")]
    check(all(not S.v2_enabled(dict(S.DEGRADE_KEYS, **S.DEGRADE_PRESETS[p])) for p in v1),
          f"v1 presets {v1} do not enable v2")
    check(S.v2_enabled(dict(S.DEGRADE_KEYS, **S.DEGRADE_PRESETS["tw_photo_hard"])), "tw_photo_hard enables v2")
    sp = S.make_spec("typewriter", preset="photo", seed=11)
    check("pipeline" not in S.generate(sp)["truth"], "a v1 page carries no v2 truth fields")


def t_v2_determinism() -> None:
    print("6b. v2 determinism")
    spec = S.make_spec("typewriter", preset="tw_photo_hard", seed=7)
    a, b = S.generate(spec), S.generate(spec)
    check(a["truth"]["image"]["sha256"] == b["truth"]["image"]["sha256"], "hard preset: same image hash")
    check(a["truth"]["truth_sha256"] == b["truth"]["truth_sha256"], "hard preset: same truth hash")
    c = S.generate(S.make_spec("typewriter", preset="tw_photo_hard", seed=8))
    check(c["truth"]["image"]["sha256"] != a["truth"]["image"]["sha256"], "hard preset: different seed, different page")


BASE_UNEXPLAINED: list = []


def _ink_box_swallow(res) -> float:
    """Share of words whose ink lies >10% inside ANOTHER word's ink box
    (ink_aabb, clean geometry). Layout boxes include side bearings, so for
    touching words the layout-box swallow test of smoke 1a is not meaningful;
    the ink box is the grading truth (CLAUDE.md)."""
    t = res["truth"]
    H, W = res["idmap_clean"].shape
    ink = T._ink_index(res["idmap_clean"], len(t["words"]))
    boxes = {w["id"]: [w["ink_aabb"][0] * W, w["ink_aabb"][1] * H, w["ink_aabb"][2] * W, w["ink_aabb"][3] * H]
             for w in t["words"] if w["ink_aabb"]}
    bad = 0
    for wid, (ys, xs) in ink.items():
        cx, cy = xs + 0.5, ys + 0.5
        for vid, b in boxes.items():
            if vid == wid or b[2] < cx.min() or b[0] > cx.max() or b[3] < cy.min() or b[1] > cy.max():
                continue
            if ((cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])).mean() > 0.10:
                bad += 1
                break
    return bad / max(1, len(ink))


def _ink_checks(label, res, clean_geometry: bool, touching: bool = False) -> None:
    if clean_geometry:
        c = T.check_clean(res)
        if touching:
            n_cov_fail = sum(1 for r in c["failures"] if r.get("cov2", 0) < T.COV_MIN)
            check(n_cov_fail == 0, f"{label}: every word's ink inside its (moved) layout box ({n_cov_fail} failures)")
            sw = _ink_box_swallow(res)
            check(sw <= 0.01, f"{label}: {100 * sw:.2f}% of words swallowed by a neighbour's ink box")
        else:
            check(c["agree_frac"] >= 0.99, f"{label}: {100 * c['agree_frac']:.2f}% of words' ink inside their layout box (clean check)")
        base = BASE_UNEXPLAINED[0] if BASE_UNEXPLAINED else 0
        check(c["image_dark_unexplained"] <= base,
              f"{label}: no new unexplained dark pixels ({c['image_dark_unexplained']}; same page without it: {base})")
    d = T.check_degraded(res)
    check(d["frac"] >= 0.99, f"{label}: {100 * d['frac']:.2f}% of words' warped ink inside warped polygon")


def t_v2_each_degradation() -> None:
    print("6c. each v2 degradation alone keeps truth exact")
    H = S.DEGRADE_PRESETS["tw_photo_hard"]
    base = S.generate(_memo("v2_base", {}), keep_arrays=True)
    BASE_UNEXPLAINED[:] = [T.check_clean(base)["image_dark_unexplained"]]   # MuPDF anti-aliasing specks, v1 too
    nb_words, nb_lines = len(base["truth"]["words"]), len(base["truth"]["lines"])
    texts = [ln["text"] for ln in base["truth"]["lines"]]

    # 1. specks
    r = S.generate(_memo("v2_specks", {k: H[k] for k in H if k.startswith("specks")}), keep_arrays=True)
    t = r["truth"]
    sp = [o for o in t["overlays"] if o["kind"] == "speck"]
    check(len(sp) >= 0.8 * H["specks"], f"specks: {len(sp)} of {H['specks']} placed and recorded")
    check(all(o["text"] is False for o in sp), "specks: recorded as non-text")
    check(len(t["words"]) == nb_words and [ln["text"] for ln in t["lines"]] == texts,
          "specks: words and lines unchanged (specks are not text)")
    Hh, Ww = r["clean"].shape
    on_text = sum(int((r["idmap_clean"][int(o["aabb"][1] * Hh):int(o["aabb"][3] * Hh) + 1,
                                        int(o["aabb"][0] * Ww):int(o["aabb"][2] * Ww) + 1] > 0).any()) for o in sp)
    check(on_text == 0, f"specks: none overlaps word ink ({on_text})")
    dark = sum(int((r["clean"][int(o["aabb"][1] * Hh):int(o["aabb"][3] * Hh) + 1,
                               int(o["aabb"][0] * Ww):int(o["aabb"][2] * Ww) + 1] < 200).any()) for o in sp)
    check(dark == len(sp), f"specks: every recorded speck is visible ink in the image ({dark}/{len(sp)})")
    o1 = [o for o in sp if o["o1_trigger"]]
    check(bool(o1) and all(o["category"] == "line_start" and o["side"] == "right" and o["gap_px"] > 0 for o in o1),
          f"specks: {len(o1)} O1 triggers, all just right of their line's first word")
    check(all(o["at_line_height"] for o in o1), "specks: O1 triggers sit at line height")
    _ink_checks("specks", r, True)

    # 2. tight word spacing
    r = S.generate(_memo("v2_tight", dict(tight_frac=0.4, tight_to=[0.0, 1.0])), keep_arrays=True)
    t = r["truth"]
    g = t["word_gaps"]
    check(len(t["words"]) == nb_words and [ln["text"] for ln in t["lines"]] == texts, "tight: same words and lines")
    ext = S.word_extents_px(r["idmap_clean"], len(t["words"]))
    agree = all(x["gap_px"] == ext[x["right_id"]][0] - ext[x["left_id"]][2] for x in g)
    check(agree, "tight: every recorded gap equals the id-map gap between the two words")
    tg = [x for x in g if x["tightened"]]
    check(bool(tg) and all(x["gap_px"] == x["target_px"] < x["gap_px_before"] for x in tg),
          f"tight: {len(tg)} tightened boundaries land exactly on their target gap")
    elig = sum(1 for x in g if x["eligible"])
    check(abs(len(tg) / max(1, elig) - 0.4) < 0.08, f"tight: tightened share {len(tg) / max(1, elig):.3f} of eligible ~ 0.4")
    check(sum(1 for x in tg if x["gap_px"] == 0) > 0, "tight: some word pairs touch (gap 0)")
    inner = {w["id"]: w["inner_gaps_px"] for w in t["words"]}
    check(all(x["o2_trigger"] == (x["gap_px"] <= max(inner[x["right_id"]] + inner[x["left_id"]] + [0])) for x in g),
          "tight: o2_trigger == (gap <= larger in-word gap of the pair)")
    check(sum(x["o2_trigger"] for x in g) > 0, f"tight: {sum(x['o2_trigger'] for x in g)} O2-trigger boundaries")
    # rendered text layer: columns well inside a gap hold no ink of the line
    TL = r["text_layer"]
    viol = 0
    for x in g:
        if x["gap_px"] is None or x["gap_px"] < 3:
            continue
        er, el = ext[x["right_id"]], ext[x["left_id"]]
        y0, y1 = min(er[1], el[1]), max(er[3], el[3])
        if TL[y0:y1, el[2] + 1:er[0] - 1].min() < 0.5:
            viol += 1
    check(viol == 0, f"tight: gaps >= 3 px are blank in the rendered text layer ({viol} violations)")
    gi = [abs(x["gap_px_image"] - x["gap_px"]) for x in g if x["gap_px_image"] is not None]
    check(max(gi) <= 1.0, f"tight: gap measured in the final image equals the clean gap (max diff {max(gi):.2f} px; no warp)")
    _ink_checks("tight", r, True, touching=True)
    r2 = S.generate(_memo("v2_tight_warp", dict(tight_frac=0.4, warp_top_deg=0.1, warp_mid_deg=2.2, warp_bottom_deg=3.3)))
    gi = sorted(abs(x["gap_px_image"] - x["gap_px"]) for x in r2["truth"]["word_gaps"] if x["gap_px_image"] is not None)
    check(gi[int(0.95 * (len(gi) - 1))] <= 3.0, f"tight+warp: image gap within 3 px of the clean gap for 95% of boundaries (p95 {gi[int(0.95 * (len(gi) - 1))]:.2f})")

    # 3. large overlays
    r = S.generate(_memo("v2_big", dict(stamps_big=1, signatures_big=1)), keep_arrays=True)
    t = r["truth"]
    bigs = [o for o in t["overlays"] if o["kind"] in ("stamp_big", "signature_big")]
    check(len(bigs) == 2 and all(o["text"] is False for o in bigs), "big overlays: stamp and signature recorded as non-text")
    st = next(o for o in bigs if o["kind"] == "stamp_big")
    check(len(st["lines_covered"]) >= 4, f"big overlays: stamp covers {len(st['lines_covered'])} lines")
    hit = [cv for o in bigs for cv in o["covers"] if cv["ink_overlap"] > 0]
    check(len(hit) > 0, f"big overlays: {len(hit)} words have ink touched by overlay ink")
    cb = [w for w in t["words"] if w["covered_by"]]
    check(len(cb) == len({cv["word_id"] for o in bigs for cv in o["covers"]}), "big overlays: per-word covered_by matches overlay covers")
    # recompute one coverage from the layers
    N, idm = r["nontext_layer"], r["idmap_clean"]
    cv = max(hit, key=lambda c: c["ink_overlap"])
    wm = idm == cv["word_id"] + 1
    om = S._dilate_bool(N < 0.8, 1)
    rec = float((wm & om).sum()) / max(1, int(wm.sum()))
    check(abs(rec - cv["ink_overlap"]) < 0.05, f"big overlays: recomputed ink_overlap {rec:.3f} ~ recorded {cv['ink_overlap']:.3f}")
    _ink_checks("big overlays", r, False)

    # 4. uneven baselines
    r = S.generate(_memo("v2_baselines", dict(baseline_jitter_px=4.0, baseline_slope_deg=1.0, baseline_curve_px=5.0)),
                   keep_arrays=True)
    tf = r["transform"]
    xs = np.random.default_rng(0).uniform(0, r["image"].shape[1], 3000)
    ys = np.random.default_rng(1).uniform(0, r["image"].shape[0], 3000)
    fx, fy = tf.forward(xs, ys)
    bx, by = tf.inverse(fx, fy)
    err = float(np.max(np.hypot(bx - xs, by - ys)))
    check(err < 1e-3, f"baselines: inverse(forward(p)) == p (max err {err:.2e} px)")
    sl = [ln["slope_deg"] for ln in r["truth"]["lines"]]
    check(np.ptp(sl) > 1.0, f"baselines: per-line slopes vary ({min(sl):.2f}..{max(sl):.2f} deg)")
    _ink_checks("baselines", r, False)
    hard = S.generate(S.make_spec("typewriter", preset="tw_photo_hard", seed=11), keep_arrays=True)
    hs = [abs(ln["slope_deg"]) for ln in hard["truth"]["lines"]]
    check(max(hs) >= 3.3, f"hard preset: steepest line {max(hs):.2f} deg (reaches/exceeds 3.3)")
    _ink_checks("hard preset", hard, False)

    # 5. typewriter texture + typed underlines
    r = S.generate(_memo("v2_texture", {k: H[k] for k in H if k.startswith("tw_")}), keep_arrays=True)
    check(r["truth"]["notes"]["n_counters_filled"] > 0, f"texture: {r['truth']['notes']['n_counters_filled']} counters filled")
    check(len(r["truth"]["words"]) == nb_words, "texture: same words")
    _ink_checks("texture", r, True)
    r = S.generate(_memo("v2_underlines", dict(typed_underlines=True, typed_underline_body=2)), keep_arrays=True)
    t = r["truth"]
    ul = [o for o in t["overlays"] if o["kind"] == "typed_underline"]
    check(len(ul) >= 3 and all(o["text"] is False for o in ul), f"underlines: {len(ul)} typed underlines recorded as non-text")
    check(any(o["under"] == "heading" for o in ul), "underlines: at least one under a heading")
    check([ln["text"] for ln in t["lines"]] == texts, "underlines: not counted as text lines (lines unchanged)")
    _ink_checks("underlines", r, True)


def t_text_pool() -> None:
    print("6d. text sources")
    import random as _r
    a = [S.pool_sentence(_r.Random(3)) for _ in range(1)]
    rng = _r.Random(3)
    sents = [S.pool_sentence(rng) for _ in range(500)]
    rng = _r.Random(3)
    check(sents == [S.pool_sentence(rng) for _ in range(500)] and a[0] == sents[0], "pool: deterministic by seed")
    uniq = len(set(sents)) / len(sents)
    check(uniq >= 0.85, f"pool: {100 * uniq:.1f}% of 500 sentences distinct")
    check(not any("{" in x or "}" in x for x in sents), "pool: no unfilled placeholders")

    def repeated_ngrams(t, n=5):
        ws = [w for ln in t["lines"] for w in ln["text"].split()]
        cnt = {}
        for i in range(len(ws) - n + 1):
            k = tuple(ws[i:i + n]); cnt[k] = cnt.get(k, 0) + 1
        return sum(1 for v in cnt.values() if v > 1)
    old = S.generate(S.make_spec("typewriter", preset="clean", seed=3))["truth"]
    new = S.generate(_memo("pool_page", {}, seed=3))["truth"]
    ro, rn = repeated_ngrams(old), repeated_ngrams(new)
    check(rn * 4 <= ro, f"pool: repeated 5-grams on a page {rn} (embedded articles text: {ro})")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("این یک بند آزمایشی برای خواندن از پرونده است.\n\nبند دوم هم چند واژه دارد تا کافی باشد.\n")
        path = f.name
    r = S.generate(S.make_spec("typewriter", preset="clean", seed=3, layout="prose", text_source=f"file:{path}",
                               name="file_src"))
    check("آزمایشی" in " ".join(ln["text"] for ln in r["truth"]["lines"]), "file: text read from a run-time path")
    try:
        S.file_paragraphs(str(S.ROOT / "books" / "hadpi-articles" / "text" / "0001.md"))
        check(False, "file: private document path refused")
    except ValueError:
        check(True, "file: private document path refused")
    except FileNotFoundError:
        check(False, "file: private document path refused before reading")


def main() -> int:
    t_determinism(); t_transform(); t_rotation_image(); t_injection(); t_manifest()
    t_v2_off_by_default(); t_v2_determinism(); t_v2_each_degradation(); t_text_pool()
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
