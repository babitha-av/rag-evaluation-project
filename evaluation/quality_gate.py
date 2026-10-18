import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DETERMINISTIC_RESULTS = (
    BASE_DIR / "evaluation" / "results" / "deterministic_results.csv"
)

RAGAS_RESULTS = (
    BASE_DIR / "evaluation" / "results" / "ragas_results.csv"
)


# Minimum acceptable semantic-quality thresholds
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "answer_correctness": 0.75,
    "context_precision": 0.90,
}


def load_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def calculate_deterministic_results(rows):
    total = len(rows)

    passed = sum(
        1
        for row in rows
        if row["overall_status"].strip().upper() == "PASS"
    )

    failed = total - passed

    failure_types = {}

    for row in rows:
        if row["overall_status"].strip().upper() != "PASS":
            failure_type = row.get("failure_type", "UNKNOWN").strip()

            failure_types[failure_type] = (
                failure_types.get(failure_type, 0) + 1
            )

    pass_rate = passed / total if total else 0.0

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "failure_types": failure_types,
    }


def calculate_ragas_averages(rows):
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
    ]

    averages = {}

    for metric in metrics:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) not in (None, "")
        ]

        averages[metric] = (
            sum(values) / len(values)
            if values
            else 0.0
        )

    return averages


def run_quality_gate():
    deterministic_rows = load_csv(DETERMINISTIC_RESULTS)
    ragas_rows = load_csv(RAGAS_RESULTS)

    deterministic = calculate_deterministic_results(
        deterministic_rows
    )

    ragas = calculate_ragas_averages(ragas_rows)

    print("\n=== RAG QUALITY GATE ===\n")

    # ---------------------------------------------------------
    # 1. Semantic quality
    # ---------------------------------------------------------

    print("Semantic Metrics")
    print("--------------------------------")

    semantic_pass = True

    for metric, threshold in THRESHOLDS.items():
        actual = ragas[metric]

        passed = actual >= threshold

        status = "PASS" if passed else "FAIL"

        print(
            f"{metric:<25} "
            f"{actual:.3f} "
            f"(threshold: {threshold:.3f}) "
            f"[{status}]"
        )

        if not passed:
            semantic_pass = False

    # ---------------------------------------------------------
    # 2. Deterministic quality
    # ---------------------------------------------------------

    print("\nDeterministic Checks")
    print("--------------------------------")

    print(
        f"Passed                  "
        f"{deterministic['passed']} / {deterministic['total']}"
    )

    print(
        f"Pass rate               "
        f"{deterministic['pass_rate']:.3f}"
    )

    print(
        f"Failures                "
        f"{deterministic['failed']}"
    )

    # ---------------------------------------------------------
    # 3. Failure classification
    # ---------------------------------------------------------

    if deterministic["failure_types"]:
        print("\nFailure Classification")
        print("--------------------------------")

        for failure_type, count in sorted(
            deterministic["failure_types"].items()
        ):
            print(
                f"{failure_type:<25} {count}"
            )

    # ---------------------------------------------------------
    # 4. Final quality gate
    # ---------------------------------------------------------

    deterministic_pass = deterministic["failed"] == 0

    overall_pass = semantic_pass and deterministic_pass

    print("\n--------------------------------")

    if overall_pass:
        print("QUALITY GATE: PASS")
    else:
        print("QUALITY GATE: FAIL")

    print("--------------------------------\n")

    return overall_pass


if __name__ == "__main__":
    success = run_quality_gate()

    # Exit code 0 = PASS
    # Exit code 1 = FAIL
    #
    # This is important for GitHub Actions later:
    # a failed quality gate will fail the CI job.

    raise SystemExit(0 if success else 1)
