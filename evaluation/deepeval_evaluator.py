import csv
from pathlib import Path

from deepeval import test_case
from deepeval import test_case
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "evaluation" / "results" / "rag_outputs.csv"

MAX_CASES = 2


def load_evaluation_cases():
    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def create_test_cases():
    dataset = load_evaluation_cases()

    excluded_categories = {
        "negative",
        "ambiguous",
        "adversarial",
    }

    test_cases = []

    for row in dataset:

        category = row["category"].strip().strip('"')

        if category in excluded_categories:
            continue

        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["ground_truth"],
            retrieval_context=[
                chunk.strip()
                for chunk in row["retrieved_context"].split("\n\n---\n\n")
                if chunk.strip()
            ],
        )

        test_cases.append(test_case)

    if MAX_CASES:

        return test_cases[:MAX_CASES]

    return test_cases


def run_evaluation():

    model = OllamaModel(
        model="llama3.2",
        base_url="http://localhost:11434",
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.5,
        model=model,
        include_reason=True,
    )

    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=model,
        include_reason=True,
    )

    test_cases = create_test_cases()

    print(f"Running DeepEval on {len(test_cases)} test cases...\n")

    results = []

    for index, test_case in enumerate(test_cases, start=1):

        print(
            f"[{index}/{len(test_cases)}] "
            f"{test_case.input}"
        )
        print("\n--- TEST CASE DEBUG ---")
        print("INPUT:", test_case.input)
        print("ACTUAL OUTPUT:", test_case.actual_output)
        print("RETRIEVAL CONTEXT:")
        for context in test_case.retrieval_context:
            print(">>>", context[:500])
        print("-----------------------\n")

        faithfulness_metric.measure(test_case)
        relevancy_metric.measure(test_case)

        results.append(
            {
                "question": test_case.input,
                "faithfulness": faithfulness_metric.score,
                "faithfulness_reason": faithfulness_metric.reason,
                "answer_relevancy": relevancy_metric.score,
                "answer_relevancy_reason": relevancy_metric.reason,
            }
        )

    output_path = (
        BASE_DIR
        / "evaluation"
        / "results"
        / "deepeval_results.csv"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fieldnames = [
            "question",
            "faithfulness",
            "faithfulness_reason",
            "answer_relevancy",
            "answer_relevancy_reason",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nDeepEval evaluation completed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
