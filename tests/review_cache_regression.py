#!/usr/bin/env python3
"""Offline regression checks for review refinement caches and chip placement.

Usage: source venv/bin/activate && python tests/review_cache_regression.py

The fake locator deliberately uses byte strings instead of a PDF.  It tests
review.py's evidence/derivation boundary without importing benchmark truth or
making a network call.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from click.testing import CliRunner  # noqa: E402

from farsi2epub import review  # noqa: E402
from farsi2epub.cli import main as cli_main  # noqa: E402
from farsi2epub.locate import Query  # noqa: E402


SCAN = {"x0": 0.1, "y0": 0.2, "x1": 0.2, "y1": 0.3, "source": "scan"}
REFINED = {
    "x0": 0.3,
    "y0": 0.4,
    "x1": 0.5,
    "y1": 0.45,
    "source": "scan_vlm",
}
REFINE_ALGORITHMS_SEEN: list[str] = []


def _fake_refine(
    _pdf, _page, _md, queries, boxes, read_strips, *, algorithm
):
    REFINE_ALGORITHMS_SEEN.append(algorithm)
    readings = read_strips([b"stable-strip-pixels"])
    hit = bool(readings and readings[0])
    return [dict(REFINED) if hit and boxes[i] is not None else None for i in range(len(queries))]


def _fake_no_match(
    _pdf, _page, _md, queries, boxes, read_strips, *, algorithm
):
    REFINE_ALGORITHMS_SEEN.append(algorithm)
    read_strips([b"stable-strip-pixels"])
    return [None for _ in queries]


def _check_cache_split() -> None:
    old_refine = review.refine_scan_boxes
    old_read = review.llm.read_strips
    REFINE_ALGORITHMS_SEEN.clear()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "source.pdf"
            pdf.write_bytes(b"pdf-v1")
            ws = SimpleNamespace(root=root, pdf_path=pdf)
            q1 = Query("الف", (0, 3))
            q2 = Query("ب", (4, 5))
            calls: list[tuple[int, str]] = []

            def paid_read(_client, strips, model, _page_no):
                calls.append((len(strips), model))
                return [["الف ب"] for _ in strips], {}, 0.01

            review.refine_scan_boxes = _fake_refine
            review.llm.read_strips = paid_read

            online = review._ScanBoxRefiner(ws, "model-a")
            assert online.algorithm == review.DEFAULT_BBOX_REFINE_ALGORITHM
            online._client = object()
            got = online.refine(1, "الف ب", [q1], [dict(SCAN)])
            assert got[0] and got[0]["source"] == "scan_vlm"
            assert calls == [(1, "model-a")]
            derived = online._load_cache()[online._key(1, "الف ب", q1)]
            assert derived["algorithm"] == review.DEFAULT_BBOX_REFINE_ALGORITHM
            assert REFINE_ALGORITHMS_SEEN[-1] == review.DEFAULT_BBOX_REFINE_ALGORITHM

            observations = json.loads(
                online.observation_cache_path.read_text(encoding="utf-8")
            )
            assert len(observations) == 1
            only = next(iter(observations.values()))
            assert only["status"] == "ok" and only["lines"] == ["الف ب"]
            assert only["model"] == "model-a" and only["image_sha256"]
            assert "algorithm" not in only

            # A new query changes the cheap derived key but reuses the paid
            # observation. replay() must not even attempt to create a client.
            replay = review._ScanBoxRefiner(ws, "model-a", offline=True)
            replay._get_client = lambda: (_ for _ in ()).throw(
                AssertionError("offline replay created a client")
            )
            got = replay.replay(1, "الف ب", [q2], [dict(SCAN)])
            assert got[0] and got[0]["source"] == "scan_vlm"
            assert calls == [(1, "model-a")]

            # Changing the derivation algorithm invalidates only the cheap
            # final box. The paid strip observation key remains identical and
            # the pilot can replay it without creating a client.
            pilot = review._ScanBoxRefiner(
                ws,
                "model-a",
                algorithm=review.locate_mod.REFINE_ALGORITHM_CONTEXT_ANCHOR,
                offline=True,
            )
            assert online._key(1, "الف ب", q1) != pilot._key(1, "الف ب", q1)
            assert online._observation_key(b"stable-strip-pixels") == (
                pilot._observation_key(b"stable-strip-pixels")
            )
            pilot._get_client = lambda: (_ for _ in ()).throw(
                AssertionError("pilot replay created a client")
            )
            got = pilot.replay(1, "الف ب", [q1], [dict(SCAN)])
            assert got[0] and got[0]["source"] == "scan_vlm"
            assert calls == [(1, "model-a")]
            pilot_entry = pilot._load_cache()[pilot._key(1, "الف ب", q1)]
            assert (
                pilot_entry["algorithm"]
                == review.locate_mod.REFINE_ALGORITHM_CONTEXT_ANCHOR
            )
            assert (
                REFINE_ALGORITHMS_SEEN[-1]
                == review.locate_mod.REFINE_ALGORITHM_CONTEXT_ANCHOR
            )

            # Review startup must keep the refiner alive without a key. Its
            # normal refine() path replays raw evidence in hard-offline mode
            # and cannot even attempt lazy client creation.
            old_load_env = review.llm.load_env
            saved_key = review.os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                review.llm.load_env = lambda: None
                startup = review._make_review_refiner(
                    ws,
                    "model-a",
                    review.locate_mod.REFINE_ALGORITHM_CONTEXT_ANCHOR,
                )
                assert startup.offline
                startup._get_client = lambda: (_ for _ in ()).throw(
                    AssertionError("offline review startup created a client")
                )
                q3 = Query("ج", (6, 7))
                got = startup.refine(1, "الف ب ج", [q3], [dict(SCAN)])
                assert got[0] and got[0]["source"] == "scan_vlm"
                assert calls == [(1, "model-a")]
            finally:
                review.llm.load_env = old_load_env
                if saved_key is not None:
                    review.os.environ["ANTHROPIC_API_KEY"] = saved_key
                else:
                    review.os.environ.pop("ANTHROPIC_API_KEY", None)

            observations_after_replay = json.loads(
                online.observation_cache_path.read_text(encoding="utf-8")
            )
            assert len(observations_after_replay) == 1

            # Model identity is part of both paid evidence and derivation.
            other_model = review._ScanBoxRefiner(ws, "model-b", offline=True)
            assert online._key(1, "الف ب", q1) != other_model._key(1, "الف ب", q1)
            got = other_model.replay(1, "الف ب", [q1], [dict(SCAN)])
            assert got == [SCAN]
            derived = json.loads(other_model.cache_path.read_text(encoding="utf-8"))
            assert other_model._key(1, "الف ب", q1) not in derived

            # The source fingerprint protects final boxes from a replaced PDF.
            before = online._key(1, "الف ب", q1)
            pdf.write_bytes(b"pdf-v2")
            after = review._ScanBoxRefiner(ws, "model-a")._key(1, "الف ب", q1)
            assert before != after
    finally:
        review.refine_scan_boxes = old_refine
        review.llm.read_strips = old_read


def _check_negative_semantics() -> None:
    old_refine = review.refine_scan_boxes
    old_read = review.llm.read_strips
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "source.pdf"
            pdf.write_bytes(b"pdf")
            ws = SimpleNamespace(root=root, pdf_path=pdf)
            q = Query("الف", (0, 3))

            # A missing/mislabeled model slot is transient: no raw negative and
            # no durable final no-match. It only settles this in-memory session.
            review.refine_scan_boxes = _fake_refine
            review.llm.read_strips = lambda *_args: ([[]], {}, 0.01)
            transient = review._ScanBoxRefiner(ws, "transient")
            transient._client = object()
            assert transient.refine(1, "الف", [q], [dict(SCAN)]) == [SCAN]
            assert not transient.observation_cache_path.exists()
            assert transient._key(1, "الف", q) not in transient._load_cache()
            assert not transient.pending(1, "الف", [q], [dict(SCAN)])
            later = review._ScanBoxRefiner(ws, "transient")
            assert later.pending(1, "الف", [q], [dict(SCAN)])

            # Parse/network exceptions follow the same non-durable path.
            def fail_read(*_args):
                raise RuntimeError("synthetic reader failure")

            review.llm.read_strips = fail_read
            failed = review._ScanBoxRefiner(ws, "exception")
            failed._client = object()
            try:
                failed.refine(1, "الف", [q], [dict(SCAN)])
            except RuntimeError as exc:
                assert "synthetic reader failure" in str(exc)
            else:
                raise AssertionError("reader exception did not propagate")
            assert failed._key(1, "الف", q) not in failed._load_cache()
            retry = review._ScanBoxRefiner(ws, "exception")
            assert retry.pending(1, "الف", [q], [dict(SCAN)])

            # Even a valid batch followed by a deterministic no-match remains
            # session-only. The callback batches queries, so batch-global
            # completeness cannot prove this individual query rendered/read;
            # its raw evidence is durable and makes a later replay free.
            review.refine_scan_boxes = _fake_no_match
            review.llm.read_strips = lambda _c, strips, _m, _p: (
                [["چیزی دیگر"] for _ in strips], {}, 0.01
            )
            no_match = review._ScanBoxRefiner(ws, "no-match")
            no_match._client = object()
            assert no_match.refine(1, "الف", [q], [dict(SCAN)]) == [SCAN]
            assert no_match._key(1, "الف", q) not in no_match._load_cache()
            assert not no_match.pending(1, "الف", [q], [dict(SCAN)])
            again = review._ScanBoxRefiner(ws, "no-match", offline=True)
            assert again.pending(1, "الف", [q], [dict(SCAN)])
            assert again.replay(1, "الف", [q], [dict(SCAN)]) == [SCAN]
            assert not again.pending(1, "الف", [q], [dict(SCAN)])

            # A locator/render path that never asks for a strip has literally
            # zero evidence. It must not create the old unsafe durable shape.
            review.refine_scan_boxes = (
                lambda _pdf, _page, _md, queries, _boxes, _reader, *, algorithm:
                [None for _ in queries]
            )
            zero = review._ScanBoxRefiner(ws, "zero-evidence", offline=True)
            assert zero.replay(1, "الف", [q], [dict(SCAN)]) == [SCAN]
            assert zero._key(1, "الف", q) not in zero._load_cache()
            assert not zero.observation_cache_path.exists() or not any(
                entry.get("model") == "zero-evidence"
                for entry in json.loads(
                    zero.observation_cache_path.read_text(encoding="utf-8")
                ).values()
            )

            # Do not continue trusting a v4 negative written by the unsafe
            # batch-global implementation before this guard was added.
            unsafe_key = zero._key(1, "الف", q)
            zero.cache_path.write_text(
                json.dumps({unsafe_key: {"status": "no_match", "box": None}}),
                encoding="utf-8",
            )
            unsafe = review._ScanBoxRefiner(ws, "zero-evidence", offline=True)
            assert unsafe.pending(1, "الف", [q], [dict(SCAN)])
    finally:
        review.refine_scan_boxes = old_refine
        review.llm.read_strips = old_read


def _check_observation_concurrency() -> None:
    """One paid raw read serves concurrent derivations; failures wake waiters."""
    old_refine = review.refine_scan_boxes
    old_read = review.llm.read_strips
    try:
        def run_case(*, fail: bool) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pdf = root / "source.pdf"
                pdf.write_bytes(b"pdf")
                ws = SimpleNamespace(root=root, pdf_path=pdf)
                refiner = review._ScanBoxRefiner(ws, "concurrent")
                refiner._client = object()

                api_started = threading.Event()
                second_locator_entered = threading.Event()
                release_api = threading.Event()
                duplicate_api_call = threading.Event()
                call_lock = threading.Lock()
                locator_calls = 0
                api_calls = 0

                def concurrent_refine(
                    _pdf, _page, _md, queries, boxes, read_strips, *, algorithm
                ):
                    nonlocal locator_calls
                    with call_lock:
                        locator_calls += 1
                        if locator_calls == 2:
                            second_locator_entered.set()
                    readings = read_strips([b"identical-strip-pixels"])
                    hit = bool(readings and readings[0])
                    return [
                        dict(REFINED) if hit and boxes[i] is not None else None
                        for i in range(len(queries))
                    ]

                def paid_read(_client, strips, _model, _page_no):
                    nonlocal api_calls
                    with call_lock:
                        api_calls += 1
                        if api_calls > 1:
                            duplicate_api_call.set()
                    api_started.set()
                    assert release_api.wait(2), "test did not release synthetic API"
                    if fail:
                        raise RuntimeError("synthetic concurrent reader failure")
                    return [["الف"] for _ in strips], {}, 0.01

                review.refine_scan_boxes = concurrent_refine
                review.llm.read_strips = paid_read
                outcomes: list[list[dict] | Exception] = []

                def worker(query: Query, md: str) -> None:
                    try:
                        outcomes.append(
                            refiner.refine(1, md, [query], [dict(SCAN)])
                        )
                    except Exception as exc:
                        outcomes.append(exc)

                first = threading.Thread(
                    target=worker, args=(Query("الف", (0, 3)), "الف")
                )
                second = threading.Thread(
                    target=worker, args=(Query("ب", (0, 1)), "ب")
                )
                first.start()
                assert api_started.wait(2), "synthetic API was never entered"
                second.start()
                assert second_locator_entered.wait(2), "second derivation never entered"
                # Under the old implementation the second API call starts
                # immediately while the first is blocked. A short event wait
                # makes that failure deterministic without a network call.
                assert not duplicate_api_call.wait(0.2), "duplicate raw read was billed"
                release_api.set()
                first.join(2)
                second.join(2)
                assert not first.is_alive() and not second.is_alive(), (
                    "observation waiter deadlocked"
                )
                assert api_calls == 1
                assert not refiner._observation_inflight

                if fail:
                    assert len(outcomes) == 2
                    assert sum(isinstance(value, RuntimeError) for value in outcomes) == 1
                    assert sum(value == [SCAN] for value in outcomes if isinstance(value, list)) == 1
                    assert not refiner.observation_cache_path.exists()
                    assert not refiner._load_cache()
                else:
                    assert outcomes == [[REFINED], [REFINED]]
                    assert len(refiner._load_observation_cache()) == 1
                    assert len(refiner._load_cache()) == 2

        run_case(fail=False)
        run_case(fail=True)
    finally:
        review.refine_scan_boxes = old_refine
        review.llm.read_strips = old_read


def _check_legacy_and_ui() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf = root / "source.pdf"
        pdf.write_bytes(b"pdf")
        ws = SimpleNamespace(root=root, pdf_path=pdf)
        q = Query("الف", (0, 3))
        refiner = review._ScanBoxRefiner(ws, "new-model", offline=True)
        legacy_key = refiner._legacy_key(1, "الف", q)
        refiner.cache_path.write_text(
            json.dumps({legacy_key: {"box": REFINED, "cost": 0.0}}),
            encoding="utf-8",
        )

        # Default v4 reads never silently inherit v3 geometry.
        assert refiner.apply_cached(1, "الف", [q], [dict(SCAN)]) == [SCAN]
        old = refiner.apply_legacy_cached(1, "الف", [q], [dict(SCAN)])
        assert old[0] and old[0]["source"] == "scan_vlm"
        compat = review._ScanBoxRefiner(
            ws, "new-model", offline=True, read_legacy=True
        )
        assert compat.apply_cached(1, "الف", [q], [dict(SCAN)])[0] == REFINED
        assert not compat.pending(1, "الف", [q], [dict(SCAN)])
        helper = review._ScanBoxRefiner.for_legacy_regrade(ws, "new-model")
        assert helper.apply_cached(1, "الف", [q], [dict(SCAN)])[0] == REFINED

    template = review._PAGE_TEMPLATE
    assert 'class="placement-row hunk-placement-row"' in template
    assert 'class="placement-row issue-placement-row"' in template
    assert template.index('class="hunk-actions"') < template.index(
        'class="placement-row hunk-placement-row"'
    )
    assert 'role="status" tabindex="0"' in template
    assert "el.setAttribute('aria-label', el.title)" in template


def _check_algorithm_contract_and_cli() -> None:
    assert review.BBOX_REFINE_ALGORITHMS == (
        review.locate_mod.REFINE_ALGORITHM_LEGACY,
        review.locate_mod.REFINE_ALGORITHM_CONTEXT_ANCHOR,
    )
    assert (
        review.DEFAULT_BBOX_REFINE_ALGORITHM
        == review.locate_mod.REFINE_ALGORITHM_LEGACY
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf = root / "source.pdf"
        pdf.write_bytes(b"pdf")
        ws = SimpleNamespace(root=root, pdf_path=pdf)
        try:
            review._ScanBoxRefiner(ws, "model", algorithm="not-an-algorithm")
        except ValueError as exc:
            assert "unknown bbox refine algorithm" in str(exc)
        else:
            raise AssertionError("invalid algorithm was accepted")

    help_result = CliRunner().invoke(cli_main, ["review", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "--bbox-refine-algorithm" in help_result.output
    assert "legacy_v1" in help_result.output
    assert "context_anchor_v1" in help_result.output
    assert "default: legacy_v1" in help_result.output


def main() -> None:
    _check_cache_split()
    _check_negative_semantics()
    _check_observation_concurrency()
    _check_legacy_and_ui()
    _check_algorithm_contract_and_cli()
    print("review cache/UI regression: OK")


if __name__ == "__main__":
    main()
