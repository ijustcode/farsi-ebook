"""Held-out test of Gemini 4 Carbon as the first (whole-page, Markdown) reader.

  ./venv/bin/python tests/heldout_first_pass.py dry     # $0, no network: inputs, key yes/no, plan, cap
  ./venv/bin/python tests/heldout_first_pass.py read    # PAID (Google): hadpi p3-p5 x 2 runs, cached
  ./venv/bin/python tests/heldout_first_pass.py score   # $0: frozen scorer, owner's audited truth

Gemini was made the default first reader (D23) on development pages only
(tests/gemini_first_pass_eval.py). This is the held-out test: the same call,
`llm.transcribe_page(None, png, config.MODEL_HARD_GEMINI, page, page_count,
title_fa)` on the hi-res page render, exactly as `transcribe` makes it, on
hadpi-articles p3, p4, p5, two runs each. `read` writes the freeze record
before its first call and refuses to run if the configuration has changed
since. A cached reading is never requested again; a failed page is recorded
and not requested again either (no fallback to another model, no retry beyond
what transcribe_page does). Scoring uses the frozen scorer exactly as for arm
`current` in heldout_results.json (Markdown, prefer "md"). Readings and results
stay under out/ (page text is private and is never printed).

CRITERIA
  Fixed 2026-11-15 before any held-out whole-page Gemini reading existed (owner rule D25).
  Scorer: frozen copy (heldout_score.load_frozen). Truth: owner-audited hadpi p3-p5.
  Baseline on the same pages and scorer: Sonnet whole page 145/1097 = 13.2% lenient,
  393/1163 = 33.8% strict (one run).
  CONFIRMED  : lenient errors pooled over both runs <= 6.6% of truth words
               AND each run separately (pooled over the three pages) <= 6.6%
               AND strict errors pooled over both runs <= 33.8%.
  WARNING    : lenient pooled over both runs >= 13.2%, or either run >= 13.2%.
  INCONCLUSIVE: anything else.
  A missing reading (failed or refused page) means NOT JUDGED; the failure itself is reported.
  Confidence is reported beside each page's real error, with no criterion: three pages
  cannot calibrate a cutoff.
  The test is run once. Nothing may be tuned on these pages afterwards.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from farsi2epub import config, hard_reader, llm  # noqa: E402

# Every output path derives from OUT at call time, so a test can point it elsewhere.
OUT = ROOT / "out" / "transcription_experiments"
STORED = ROOT / "out" / "transcription_experiments"     # comparison rows (read only)
SLUG = "hadpi-articles"
PAGES = {"h3": 3, "h4": 4, "h5": 5}
PIDS = ("h3", "h4", "h5")
REPS = (0, 1)
PLAN = [(pid, rep) for rep in REPS for pid in PIDS]       # rep 0 for p3-p5, then rep 1
CAP_USD = 0.30
CALL_RESERVE_USD = 0.05
LENIENT_OK, LENIENT_WARN, STRICT_OK = 0.066, 0.132, 0.338
BASELINE = {"lenient": (145, 1097), "strict": (393, 1163)}

_doc = __doc__ or ""
CRITERIA = _doc[_doc.index("  Fixed 2026-11-15"):_doc.index("afterwards.") + len("afterwards.")]


def cache_dir() -> Path:
    return OUT / "gemini_first_pass_heldout"


def freeze_path() -> Path:
    return OUT / "heldout_freeze_first_pass.json"


def ledger_path() -> Path:
    return OUT / "ledger_heldout_first_pass.jsonl"


def status_path() -> Path:
    return OUT / "heldout_first_pass_status.json"


def results_path() -> Path:
    return OUT / "heldout_results_first_pass.json"


def _sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _git_head() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _png_path(pid: str) -> Path:
    return ROOT / "books" / SLUG / "pages" / "hires" / f"{PAGES[pid]:04d}.png"


def _meta() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "books" / SLUG / "book.yaml").read_text(encoding="utf-8"))


def _png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def configuration() -> dict:
    """Everything that decides what the held-out request is; read from the code."""
    meta = _meta()
    thinking = inspect.signature(hard_reader.gemini_items_body).parameters["thinking_level"].default
    gemini_src = "\n".join(inspect.getsource(f) for f in (
        hard_reader.gemini_items_body, hard_reader.gemini_call, hard_reader._gemini_post,
        hard_reader.gemini_parse, hard_reader._inline_schema))
    return {
        "model": config.MODEL_HARD_GEMINI,
        "call": "llm.transcribe_page(None, png, config.MODEL_HARD_GEMINI, page, page_count, title_fa)",
        "transcribe_system_sha256": _sha(llm.TRANSCRIBE_SYSTEM),
        "page_transcription_schema_sha256": _sha(json.dumps(llm.PageTranscription.model_json_schema(), sort_keys=True)),
        "dev_script_sha256": _sha((HERE / "gemini_first_pass_eval.py").read_bytes()),
        "transcribe_page_source_sha256": _sha(inspect.getsource(llm.transcribe_page)),
        "gemini_request_source_sha256": _sha(gemini_src),
        "gemini": {"url": hard_reader.GEMINI_URL, "image_resolution": hard_reader.GEMINI_RESOLUTION,
                   "thinking_level": thinking, "max_output_tokens": llm.MAX_OUTPUT_TOKENS,
                   "attempts": hard_reader.GEMINI_ATTEMPTS},
        "input": {"slug": SLUG, "render": "pages/hires", "page_count": meta["page_count"],
                  "title_hint_sha256": _sha(str(meta.get("title_fa") or "")),
                  "png_sha256": {pid: _sha(_png_path(pid).read_bytes()) for pid in PIDS}},
        "reps": list(REPS),
        "criteria": CRITERIA,
    }


def config_hash(cfg: dict) -> str:
    return _sha(json.dumps(cfg, sort_keys=True, ensure_ascii=False))


def ledger_total() -> float:
    p = ledger_path()
    if not p.is_file():
        return 0.0
    return sum(float(json.loads(l).get("cost") or 0.0) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())


def _append_ledger(entry: dict) -> None:
    with ledger_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _status() -> dict:
    p = status_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"failed": []}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _failed_keys(st: dict) -> set:
    return {(f["pid"], f["rep"]) for f in st.get("failed", [])}


# ---------------------------------------------------------------------------

def cmd_dry() -> None:
    import heldout_score as HS
    print("Held-out first-pass test (Gemini whole page), dry run: no network")
    ok = True
    for pid in PIDS:
        p = _png_path(pid)
        if not p.is_file():
            print(f"  {pid}: hi-res PNG MISSING ({p.relative_to(ROOT)})")
            ok = False
            continue
        data = p.read_bytes()
        w, h = _png_size(data)
        print(f"  {pid}: {p.relative_to(ROOT)}  {w}x{h} px  {len(data):,} bytes")
    try:
        HS.audited_or_exit()
        print("  truth audited (final exports, SHA match): yes")
    except SystemExit as exc:
        print(f"  truth audited: NO ({exc})")
        ok = False
    print(f"  Gemini key present: {'yes' if hard_reader.gemini_key() else 'no'}")
    if not ok:
        raise SystemExit("dry: inputs incomplete")
    cfg = configuration()
    h = config_hash(cfg)
    print(f"  model: {cfg['model']}  image resolution: {cfg['gemini']['image_resolution']}  "
          f"thinking: {cfg['gemini']['thinking_level']}  max_output_tokens: {cfg['gemini']['max_output_tokens']}")
    print(f"  configuration hash (frozen by read): {h}")
    fp = freeze_path()
    if fp.is_file():
        same = json.loads(fp.read_text())["configuration"] == cfg
        print(f"  existing freeze record: {'matches' if same else 'DIFFERS (read would refuse)'}")
    else:
        print("  existing freeze record: none (read writes it before the first call)")
    failed = _failed_keys(_status())
    print("  planned calls (in order):")
    for pid, rep in PLAN:
        state = ("cached" if (cache_dir() / f"{pid}_{rep}.json").is_file()
                 else "failed earlier, not requested again" if (pid, rep) in failed else "to request")
        print(f"    {pid} rep{rep}: {state}")
    print(f"  spend cap ${CAP_USD:.2f} (stop before a call when ledger + ${CALL_RESERVE_USD:.2f} > cap); "
          f"ledger so far ${ledger_total():.4f}; price now {config.gemini_price()} per M tokens")


def cmd_read() -> None:
    import transcription_experiments as TE
    TE._HELDOUT_RELEASED = True            # as heldout_vendor.py; the request below does not go through TE
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    fp = freeze_path()
    if fp.is_file():
        if json.loads(fp.read_text())["configuration"] != cfg:
            raise SystemExit("configuration changed since heldout_freeze_first_pass.json; refusing")
    else:
        _write_atomic(fp, json.dumps({
            "frozen_at": _now(), "git_head": _git_head(), "configuration": cfg,
            "configuration_sha256": config_hash(cfg),
            "script_sha256": _sha(Path(__file__).read_bytes()),
            "price_usd_per_mtok_at_freeze": config.gemini_price(),
            "cap_usd": CAP_USD, "criteria": CRITERIA}, ensure_ascii=False, indent=1))
    key = hard_reader.gemini_key()
    meta = _meta()
    cache_dir().mkdir(parents=True, exist_ok=True)
    st = _status()
    for pid, rep in PLAN:
        path = cache_dir() / f"{pid}_{rep}.json"
        if path.is_file():
            print(f"{pid} rep{rep}: cached, not requested")
            continue
        if (pid, rep) in _failed_keys(st):
            print(f"{pid} rep{rep}: failed earlier, not requested again")
            continue
        total = ledger_total()
        if total + CALL_RESERVE_USD > CAP_USD:
            print(f"STOP: spend cap: ledger ${total:.4f} + ${CALL_RESERVE_USD:.2f} > ${CAP_USD:.2f}")
            return
        page = PAGES[pid]
        png = _png_path(pid).read_bytes()
        try:
            result, usage, cost = llm.transcribe_page(None, png, config.MODEL_HARD_GEMINI, page,
                                                      meta["page_count"], meta.get("title_fa"))
        except Exception as exc:  # noqa: BLE001  every failure is recorded; none falls back
            cost = float(getattr(exc, "cost", 0.0) or 0.0)
            err = TE.redact(str(exc), key or "")
            rec = {"ts": _now(), "pid": pid, "rep": rep, "ok": False, "cost": cost,
                   "usage": getattr(exc, "usage", None), "error_class": type(exc).__name__, "error": err}
            _append_ledger(rec)
            st.setdefault("failed", []).append({k: rec[k] for k in ("ts", "pid", "rep", "cost", "error_class", "error")})
            st["updated_at"] = _now()
            _write_atomic(status_path(), json.dumps(st, ensure_ascii=False, indent=1))
            print(f"{pid} rep{rep}: FAILED {type(exc).__name__}: {err} (billed ${cost:.4f}; "
                  f"running ${ledger_total():.4f})", flush=True)
            if isinstance(exc, hard_reader.ReaderUnavailable):
                print("STOP: Gemini unavailable (key, access or billing); no further page requested")
                return
            continue
        _write_atomic(path, json.dumps({"pid": pid, "rep": rep, "parsed": result.model_dump(), "usage": usage,
                                        "cost": cost, "timestamp": _now(), "model": config.MODEL_HARD_GEMINI},
                                       ensure_ascii=False, indent=1))
        _append_ledger({"ts": _now(), "pid": pid, "rep": rep, "ok": True, "cost": cost, "usage": usage,
                        "error_class": None, "error": None})
        print(f"{pid} rep{rep} in={usage['input_tokens']} out={usage['output_tokens']} ${cost:.4f} "
              f"running ${ledger_total():.4f}", flush=True)
    print(f"done; ledger total ${ledger_total():.4f}")


def _cell(e: float, n: int) -> str:
    return f"{e:g}/{n} {100 * e / n:4.1f}%"


def _pool(rows: list, folding: str) -> tuple[float, int]:
    return (sum(r[folding]["errors"] for r in rows), sum(r[folding]["n_truth"] for r in rows))


def cmd_score() -> None:
    import heldout_score as HS
    truth_info = HS.audited_or_exit()
    FZ = HS.load_frozen()                  # frozen scorer; also blocks Anthropic calls
    folds = FZ.FOLDINGS
    st = _status()
    missing, readings = [], {}
    for pid, rep in PLAN:
        f = cache_dir() / f"{pid}_{rep}.json"
        if f.is_file():
            readings[(pid, rep)] = json.loads(f.read_text(encoding="utf-8"))
        else:
            missing.append([pid, rep])

    first = json.loads((STORED / "heldout_results.json").read_text())
    second = json.loads((STORED / "heldout_results_2b.json").read_text())
    stored_rows = {pid: (("sonnet whole pg", first["pages"][pid]["current"]), ("t1r_opus bands", first["pages"][pid]["t1r_opus"]),
                         ("gg_carbon bands", second["pages"][pid])) for pid in PIDS}
    hdr = f"{'reader':16s} {'page':4s} {'run':3s} " + " ".join(f"{x:>14s}" for x in folds) + f" {'invented':>8s} {'conf':>5s}  cost"
    print(hdr)
    rows = {}
    for pid in PIDS:
        for rep in REPS:
            r = readings.get((pid, rep))
            if r is None:
                print(f"{'gemini whole pg':16s} {pid:4s} {rep:<3d} (missing)")
                continue
            sc = FZ.score_output(pid, {"md": r["parsed"]["text_md"], "lines": None}, "md")   # as arm `current`
            row = {"pid": pid, "rep": rep, "confidence": r["parsed"].get("confidence"), "cost": r["cost"],
                   "usage": r.get("usage"), **sc}
            rows[(pid, rep)] = row
            print(f"{'gemini whole pg':16s} {pid:4s} {rep:<3d} " + " ".join(f"{_cell(row[x]['errors'], row[x]['n_truth']):>14s}" for x in folds)
                  + f" {row['invented']:8d} {row['confidence']:5.2f}  ${row['cost']:.4f}")
        for name, src in stored_rows[pid]:
            for row in src:
                print(f"{name:16s} {pid:4s} {row['rep']:<3d} " + " ".join(f"{_cell(row[x]['errors'], row[x]['n_truth']):>14s}" for x in folds)
                      + f" {row['invented']:8d} {'':5s}  " + (f"${row['cost']:.4f}" if row.get("cost") is not None else ""))

    comp = {"current (sonnet whole page)": [r for pid in PIDS for r in first["pages"][pid]["current"]],
            "t1r_opus (bands)": [r for pid in PIDS for r in first["pages"][pid]["t1r_opus"]],
            "gg_carbon (bands)": [r for pid in PIDS for r in second["pages"][pid]]}
    print("\nPooled over the three pages (errors / truth words):")
    pooled = {"gemini whole page": {}}
    for rep in REPS:
        rr = [rows[(p, rep)] for p in PIDS if (p, rep) in rows]
        if rr:
            pooled["gemini whole page"][f"run {rep}"] = {x: _pool(rr, x) for x in folds}
            print(f"  {'gemini whole pg':28s} run {rep} ({len(rr)} pages) "
                  + "  ".join(f"{x} {_cell(*pooled['gemini whole page'][f'run {rep}'][x])}" for x in folds))
    if rows:
        pooled["gemini whole page"]["both runs"] = {x: _pool(list(rows.values()), x) for x in folds}
        print(f"  {'gemini whole pg':28s} both ({len(rows)} readings) "
              + "  ".join(f"{x} {_cell(*pooled['gemini whole page']['both runs'][x])}" for x in folds))
    for name, rr in comp.items():
        pooled[name] = {}
        for rep in sorted({r["rep"] for r in rr}):
            sub = [r for r in rr if r["rep"] == rep]
            pooled[name][f"run {rep}"] = {x: _pool(sub, x) for x in folds}
            print(f"  {name:28s} run {rep} ({len(sub)} pages) "
                  + "  ".join(f"{x} {_cell(*pooled[name][f'run {rep}'][x])}" for x in folds))
        if len({r["rep"] for r in rr}) > 1:
            pooled[name]["both runs"] = {x: _pool(rr, x) for x in folds}
            print(f"  {name:28s} both ({len(rr)} readings) "
                  + "  ".join(f"{x} {_cell(*pooled[name]['both runs'][x])}" for x in folds))
    base = pooled["current (sonnet whole page)"]["run 0"]
    for x, (e, n) in BASELINE.items():
        if tuple(base[x]) != (e, n):
            print(f"WARNING: stored Sonnet baseline {x} is {base[x][0]:g}/{base[x][1]}, criteria state {e}/{n}")

    if st.get("failed"):
        print("\nFailures recorded by read:")
        for f in st["failed"]:
            print(f"  {f['pid']} rep{f['rep']}: {f['error_class']}: {f['error']} (billed ${f['cost']:.4f})")
    if missing:
        print(f"\nmissing readings: {missing}")
        print("NOT JUDGED")
        raise SystemExit(1)

    both = pooled["gemini whole page"]["both runs"]
    runs = [pooled["gemini whole page"][f"run {r}"] for r in REPS]

    def rate(en):
        return en[0] / en[1]
    len_both, strict_both = rate(both["lenient"]), rate(both["strict"])
    len_runs = [rate(r["lenient"]) for r in runs]
    eps = 1e-12
    if len_both <= LENIENT_OK + eps and all(x <= LENIENT_OK + eps for x in len_runs) and strict_both <= STRICT_OK + eps:
        verdict = "CONFIRMED"
    elif len_both >= LENIENT_WARN - eps or any(x >= LENIENT_WARN - eps for x in len_runs):
        verdict = "WARNING"
    else:
        verdict = "INCONCLUSIVE"
    print("\nCriteria (fixed before any reading):")
    print(f"  lenient both runs {100 * len_both:.1f}% (<= 6.6%?)  per run "
          + ", ".join(f"{100 * x:.1f}%" for x in len_runs) + f"  strict both runs {100 * strict_both:.1f}% (<= 33.8%?)")
    print(f"VERDICT: {verdict}")
    res = {"scored_at": _now(), "truth": truth_info,
           "frozen_scorer_sha256": json.loads(HS.FREEZE.read_text())["script"]["sha256"],
           "freeze": json.loads(freeze_path().read_text()) if freeze_path().is_file() else None,
           "readings": [{k: v for k, v in r.items()} for r in rows.values()],
           "pooled": {k: {kk: {x: list(e) for x, e in vv.items()} for kk, vv in v.items()} for k, v in pooled.items()},
           "failed": st.get("failed", []), "ledger_total_usd": ledger_total(),
           "criteria": CRITERIA, "verdict": verdict}
    _write_atomic(results_path(), json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"results: {results_path().relative_to(ROOT) if results_path().is_relative_to(ROOT) else results_path()}")


if __name__ == "__main__":
    cmds = {"dry": cmd_dry, "read": cmd_read, "score": cmd_score}
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg not in cmds:
        raise SystemExit(f"usage: {Path(__file__).name} {{{','.join(cmds)}}}")
    cmds[arg]()
