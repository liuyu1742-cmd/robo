"""Summarize frozen blind-test failures without changing the test cases."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


def failure_reason(case: Mapping) -> str:
    """Return one stable, user-facing category for a failed raw prediction."""
    error = str(case.get("error") or "")
    lowered = error.lower()
    if "keyerror" in lowered or "catalog" in lowered:
        return "catalog_compatibility"
    if "ambiguous" in lowered or "请说明" in error:
        return "ambiguous_instruction"
    if error:
        return "runtime_error"
    return "sequence_mismatch"


def summarize_failures(cases: Sequence[Mapping]) -> dict[str, object]:
    """Count only raw-sequence failures by reason, task and object."""
    failures = [case for case in cases if not bool(case.get("raw_exact"))]
    return {
        "total_cases": len(cases),
        "failures": len(failures),
        "by_reason": dict(sorted(Counter(failure_reason(case) for case in failures).items())),
        "by_task": dict(
            sorted(Counter(str(case.get("acceptance_task", "unknown")) for case in failures).items())
        ),
        "by_object": dict(
            sorted(Counter(str(case.get("object", "unknown")) for case in failures).items())
        ),
    }
