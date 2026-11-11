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
  5. tracked manifest (tests/data/synth_manifest.json): two entries regenerate
     to the recorded hashes, when the font files and PyMuPDF version match
     (otherwise reported as SKIP, not as a pass).

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
    picks = [e for e in man["entries"] if e["name"].startswith(("modern_naskh", "typewriter")) and "photo" in e["name"]][:2]
    for e in picks:
        drift = [f["path"] for f in e["fonts"] if not Path(S._abs(f["path"])).exists()
                 or S.sha256_file(S._abs(f["path"])) != f["sha256"]]
        if drift:
            print(f"  SKIP {e['name']}: font file changed or missing: {drift}"); continue
        res = S.generate(e["spec"])
        check(res["truth"]["image"]["sha256"] == e["image_sha256"], f"{e['name']}: image hash matches manifest")
        check(res["truth"]["truth_sha256"] == e["truth_sha256"], f"{e['name']}: truth hash matches manifest")


def main() -> int:
    t_determinism(); t_transform(); t_rotation_image(); t_injection(); t_manifest()
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
