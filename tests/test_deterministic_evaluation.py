import csv
from pathlib import Path

import pytest

from evaluation.deterministic_checks import evaluate_row


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "evaluation" / "results" / "rag_outputs.csv"


def load_rag_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# These are known defects identified by the deterministic evaluation.
# They are intentionally tracked as expected failures.
KNOWN_FAILURES = {
    "TC007": "Retrieval failure: security.md was not retrieved.",
    "TC015": "Reasoning failure: refund eligibility after 20 days.",
    "TC016": "Ambiguity handling failure: model answered without clarification.",
    "TC021": "Reasoning failure: refund eligibility after 20 days.",
}


@pytest.mark.parametrize(
    "row",
    load_rag_results(),
    ids=lambda row: row["id"],
)
def test_deterministic_evaluation(row):

    result = evaluate_row(row)

    test_id = row["id"]

    if test_id in KNOWN_FAILURES:
        pytest.xfail(
            reason=KNOWN_FAILURES[test_id]
        )

    assert result["overall_status"] == "PASS", (
        f"{test_id} failed: "
        f"{result['failure_type']} - "
        f"{result['failure_reason']}"
    )
