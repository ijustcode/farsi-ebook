"""Offline checks that reader budget reservations are realistic upper bounds.

Reservations used to treat base64 PNG length as input tokens, which made a
4-crop batch reserve dollars while costing cents. They now come from pixel
dimensions under the documented visual-token rule. No API call is made.
"""
from pathlib import Path
import random
import sys
import tempfile
import threading
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz
from farsi2epub import llm
from farsi2epub.config import PRICES, MODEL_STRONG, MAX_IMAGE_TOKENS, image_tokens
from farsi2epub.scan import (Budget, PlacementService, reader_reservation, _png_size,
                             RESERVATION_MARGIN, RESERVATION_OVERHEAD_TOKENS)

P = PRICES[MODEL_STRONG]


def png(w, h, noise=False, seed=0):
    """A real PNG of w x h; noise makes it as incompressible as a colour scan."""
    if noise:
        samples = random.Random(seed).randbytes(w * h * 3)
    else:
        samples = bytes([255]) * (w * h * 3)
    return fitz.Pixmap(fitz.csRGB, w, h, samples, False).tobytes("png")


def old_byte_reservation(batch):
    # The superseded arithmetic, kept here only to show the size of the defect.
    return ((sum(len(b) * 4 / 3 for b in batch) + 16000) * P["in"]
            + llm.REGION_OUTPUT_TOKENS * P["out"]) / 1_000_000


def rule_worst_case(sizes, batch_len):
    """Worst-case cost from the documented rule alone: visual tokens + a
    byte bound on prompt/labels + the full output cap (no allowance, no margin)."""
    text = len(llm.READ_REGIONS_SYSTEM.encode()) + sum(len(f"Region {i}:") for i in range(batch_len))
    tokens = sum(image_tokens(w, h) for w, h in sizes) + text
    return (tokens * P["in"] + llm.REGION_OUTPUT_TOKENS * P["out"]) / 1_000_000


def run():
    # The token rule reproduces the documented high-resolution examples
    # (platform docs, vision "Resolution and token cost").
    for (w, h), tokens in {(200, 200): 64, (1000, 1000): 1296, (1092, 1092): 1521,
                           (1920, 1080): 2691, (2000, 1500): 3888, (3840, 2160): 4784}.items():
        assert image_tokens(w, h) == tokens, (w, h, image_tokens(w, h))
    # Elongated crops beyond the 2576 px edge are downscaled, never billed raw.
    assert image_tokens(3864, 60) <= -(-3864 // 28) * -(-60 // 28)
    assert MAX_IMAGE_TOKENS == 4784 and image_tokens(8000, 8000) == 4761  # 1932x1932 after resize
    assert all(image_tokens(w, h) <= MAX_IMAGE_TOKENS for w in (1, 999, 2576, 9000) for h in (1, 77, 3000, 9000))

    # Dimensions come from the PNG header, matching what the crop renderer emits.
    with fitz.open() as doc:
        page = doc.new_page(width=420, height=600)
        crop = PlacementService._render(page, fitz.Rect(10, 100, 410, 115), 1.5)
        pix = page.get_pixmap(matrix=fitz.Matrix(2576 / 600 * 1.5, 2576 / 600 * 1.5),
                              clip=fitz.Rect(10, 100, 410, 115), alpha=False)
        assert _png_size(crop) == (pix.width, pix.height)
    assert _png_size(b"pixels") is None and _png_size(b"") is None

    # 1. A realistic batch: four full-width line crops of a noisy colour scan
    # (hadpi-like). New reservation is near the rule's worst case and far
    # below the old byte-based figure.
    sizes = [(2393, 70), (2300, 66), (1800, 72), (640, 70)]
    batch = [png(w, h, noise=True, seed=i) for i, (w, h) in enumerate(sizes)]
    new, old, worst = reader_reservation(batch, MODEL_STRONG), old_byte_reservation(batch), rule_worst_case(sizes, 4)
    print(f"typical batch: old ${old:.4f}  new ${new:.4f}  rule worst case ${worst:.4f}  "
          f"new/worst {new / worst:.2f}  old/new {old / new:.1f}")
    assert worst <= new <= 2.5 * worst, (new, worst)
    assert old >= 20 * new, (old, new)

    # 2. Never below the worst case, for tiny, typical, edge-limited and
    # API-downscaled crops, and for unreadable dimensions.
    for size in [(1, 1), (5, 13), (28, 28), (62, 13), (2393, 70), (2576, 2576),
                 (3864, 60), (60, 3864), (2458, 3864), (8000, 8000), (12000, 40)]:
        for n in (1, 4):
            b = [png(*size)] * n if size[0] * size[1] <= 4_000_000 else None
            if b is None:  # avoid materialising giant pixmaps; header is all that is read
                header = png(1, 1)
                b = [header[:16] + size[0].to_bytes(4, "big") + size[1].to_bytes(4, "big") + header[24:]] * n
            assert _png_size(b[0]) == size
            r = reader_reservation(b, MODEL_STRONG)
            assert r >= rule_worst_case([size] * n, n), (size, n)
            assert r >= RESERVATION_MARGIN * rule_worst_case([size] * n, n)
    worst_any = rule_worst_case([(8000, 8000)], 1)
    assert reader_reservation([b"not a png"], MODEL_STRONG) >= worst_any
    assert reader_reservation([b"not a png"], MODEL_STRONG) == reader_reservation([png(8000, 1)[:8] + b"bad"], MODEL_STRONG)
    # Both configured models are bounded (the cap covers both resolution tiers).
    for model in PRICES:
        assert reader_reservation(batch, model) > 0

    # 3. hadpi-shaped run: four pages acquiring concurrently, ~20-30 noisy
    # line crops each, $5 cap. All four hold a reservation at the same time.
    pages = [[png(2393, 70 + j % 9, noise=True, seed=100 * p + j) for j in range(n)]
             for p, n in enumerate((25, 26, 27, 28))]  # 7 batches each
    def concurrent(reserve_of):
        budget, refused, barrier = Budget(5.0), [], threading.Barrier(4)
        def page(crops):
            for k in range(0, len(crops), 4):
                amount = reserve_of(crops[k:k + 4])
                barrier.wait()
                ok = budget.reserve(amount)
                barrier.wait()  # every page holds its reservation here
                if not ok:
                    refused.append(amount); continue
                budget.settle(amount, 0.006)  # roughly what such a batch costs
        threads = [threading.Thread(target=page, args=(c,)) for c in pages]
        for t in threads: t.start()
        for t in threads: t.join()
        return budget, refused
    budget, refused = concurrent(lambda b: reader_reservation(b, MODEL_STRONG))
    peak_new = 4 * max(reader_reservation(c[k:k + 4], MODEL_STRONG) for c in pages for k in range(0, len(c), 4))
    print(f"hadpi-shaped: new peak outstanding ${peak_new:.3f}, refusals {len(refused)}, "
          f"spent ${budget.spent:.3f}")
    assert not refused and budget.reserved < 1e-9 and budget.uncertain == 0
    _, refused_old = concurrent(old_byte_reservation)
    print(f"hadpi-shaped under the old byte formula: {len(refused_old)} refusals")
    assert refused_old, "the old arithmetic should reproduce the false budget_exhausted"

    # 4. A genuinely over-budget run is still refused, before any dispatch,
    # and settlement/uncertain semantics are unchanged.
    one = reader_reservation(batch, MODEL_STRONG)
    b = Budget(1.5 * one)
    assert b.reserve(one) and not b.reserve(one)
    b.settle(one, 0.01)
    assert b.snapshot()["reserved"] < 1e-12 and abs(b.spent - 0.01) < 1e-12
    b.settle(0, None)
    assert b.reserve(one)
    b.settle(one, None)  # transport failure: the whole reservation stays held
    assert abs(b.uncertain - one) < 1e-12 and not b.reserve(one)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with fitz.open() as doc:
            doc.new_page(); doc.save(root / "source.pdf")
        ws = SimpleNamespace(root=root, pdf_path=root / "source.pdf")
        calls = []
        def reader(images, *a):
            calls.append(len(images)); raise RuntimeError("simulated transport failure")
        service = PlacementService(ws, reader=reader, max_cost=0.5 * one)
        try:
            service._read(batch, 1, True)
        except RuntimeError as exc:
            assert str(exc) == "budget_exhausted"
        else:
            raise AssertionError("expected budget_exhausted")
        assert calls == [] and service.budget.snapshot()["spent"] == 0
        service.close()
        service = PlacementService(ws, reader=reader, max_cost=5)
        try:
            service._read(batch, 1, True)
        except RuntimeError as exc:
            assert str(exc) == "simulated transport failure"
        snap = service.budget.snapshot()
        assert calls == [4] and abs(snap["uncertain"] - one) < 1e-12 and snap["reserved"] < 1e-12
        assert len(list((root / "locate_evidence" / "runs").glob("*.json"))) == 1
        service.close()
    print("READER RESERVATION REGRESSIONS PASSED")


if __name__ == "__main__":
    run()
