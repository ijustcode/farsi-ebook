#!/usr/bin/env python3
"""Regression checks for the single-page audited development pilot."""
from __future__ import annotations

from copy import deepcopy
import json

import bbox_development_pilot as pilot


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main() -> int:
    fixture = json.loads(pilot.DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    pilot.validate_fixture(fixture)
    report = pilot.score_fixture(fixture)
    historical = report["treatments"]["historical_full_query"]
    consolidated = report["treatments"]["consolidated_offline"]
    check(historical["metrics"]["full_target_coverage_rate"] == .8,
          "historical control must retain eight full-coverage cases")
    check(historical["failure_categories"] == {
        "adjacent_word_only": 1, "partial_target_boundary": 1},
        "historical failures must remain explicitly diagnosed")
    check(consolidated["metrics"]["unresolved_rate"] == 1.,
          "saved consolidated offline capture must retain ten null boxes")
    check(consolidated["failure_categories"] == {"unresolved_no_saved_box": 10},
          "availability failures must not be relabeled as placement errors")
    check(report["paired_diagnostic"]["page_clusters"] == 1
          and report["paired_diagnostic"]["statistical_inference"] ==
          "unavailable_single_development_page", "pilot must not claim inference")
    check(report["cost_effect"].startswith("Replay is fully offline"),
          "pilot must retain its zero-cost contract")

    damaged = deepcopy(fixture)
    damaged["cases"][0]["target"]["word_ids"] = damaged["cases"][0]["target"]["word_ids"][:-1]
    try:
        pilot.validate_fixture(damaged)
    except ValueError as exc:
        check("integrity" in str(exc), "tamper refusal should identify integrity")
    else:
        raise AssertionError("target tampering must be refused")
    print("bbox development pilot regressions: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
