"""Re-export: the line finder now lives in ``farsi2epub/line_finder.py``.

``import line_finder`` from tests/ returns that module itself (every name,
underscore ones included, and the same module object for patching)."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import farsi2epub.line_finder as _lf  # noqa: E402

sys.modules[__name__] = _lf
