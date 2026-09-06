"""Public evidence-based placement contracts.

Live status describes evidence support, not audited accuracy. Independent
word-geometry metrics live only in tests/bbox_metrics.py; the historical v5
instrument is frozen in tests/historical/placement.py.
"""
from .page_map import PlacementResult, PrintedWord, place
from .scan import PlacementService

__all__ = ["PlacementResult", "PrintedWord", "PlacementService", "place"]
