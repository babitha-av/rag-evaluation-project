import csv
from pathlib import Path

import pytest

from evaluation.deterministic_checks import answer_contains_refusal, evaluate_row


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "evaluation" / "results" / "rag_outputs.csv"


def load_rag_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# These are known defects identified by the deterministic evaluation.
# They are intentionally tracked as expected failures.
KNOWN_FAILURES = {
    "TC007": "Retrieval failure: security.md was not retrieved.",
    "TC009": "Unsupported answer: claims mobile app access without evidence.",
    "TC015": "Reasoning failure: refund eligibility after 20 days.",
    "TC016": "Ambiguity handling failure: model answered without clarification.",
    "TC021": "Reasoning failure: refund eligibility after 20 days.",
}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("No, the context does not mention Apple Pay.", True),
        ("No, the context does not provide information on phone support.", True),
        ("I can't fulfill this request.", True),
        (
            "Yes, the context does not mention a mobile app, but users can access the web app.",
            False,
        ),
    ],
)
def test_answer_contains_refusal(answer, expected):
    assert answer_contains_refusal(answer) is expected


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
